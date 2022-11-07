#!/usr/bin/env python
from __future__ import print_function

import os
import math
import time
import glob
import wandb
import pickle
import random
import resource

from collections import OrderedDict

import apex
import torch
import torch.nn as nn
import torch.optim as optim
import torch_dct as dct
import numpy as np

from tqdm import tqdm

from args import get_parser
from loss import LabelSmoothingCrossEntropy, masked_recon_loss
from model.infogcn import InfoGCN
from utils import AverageMeter, import_class
from einops import rearrange, repeat

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ema_coeffs(k, length):
    power = np.arange(length)
    coeffs = k * np.power(np.ones_like(power)*(1-k), power)
    coeffs = coeffs[::-1]
    coeffs_expand = coeffs.reshape(1,-1,1)
    return coeffs_expand

def find_best_agreement(pred, label, T):
    pred = pred.permute(1,3,0,2) # B, T, N, Cls
    y_hat_0 = torch.argmax(pred, dim=3)[:,:,-1]
    corrects_origin = (y_hat_0 == label.unsqueeze(1)).sum(0)
    new_array = []
    for t in range(T):
        new_array.append(torch.cat((pred[:,:t,0,:],pred[:,t,:,:]), dim=1))
    after_alignment = []
    best_improve = 0
    best_correction = torch.zeros(T).to(pred.device)
    best_alpha = 0
    for a in range(1, 50):
        alpha = (1-a/50)
        diff, corrects_agree = agreement(new_array, pred, label, corrects_origin, alpha, T)
        if best_improve < diff:
            best_improve = diff
            best_correction = corrects_agree
            best_alpha = alpha
    print(best_improve, best_correction/pred.size(0), best_correction - corrects_origin)
    return best_improve, best_correction

def agreement(new_array, pred, label, corrects_origin, alpha, T=64):
    corrects_agree = torch.zeros(T).to(pred.device)
    for i, seq in enumerate(new_array):
        seq = torch.softmax(seq, -1)
        coeffs_expand = ema_coeffs(1-alpha*(i+1)/64, seq.shape[1])
        coeffs_expand = torch.from_numpy(coeffs_expand.copy()).to(pred.device).to(pred.dtype)
        EMA = (seq * coeffs_expand).sum(1)
        argmax = torch.argmax(EMA, axis=-1)
        corrects_agree[i] = (argmax == label).sum(0)
    diff = (corrects_agree - corrects_origin).sum().cpu().item()
    print(diff, (corrects_agree - corrects_origin)/pred.shape[1]*100)
    return diff, corrects_agree

class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_arg()
        self.load_model()
        A_vector = self.model.get_A(self.arg.k) if self.arg.k != 8 else None

        self.load_optimizer()
        self.load_data(A_vector)

        self.best_acc = 0
        self.best_acc_epoch = 0
        self.log_recon_loss = AverageMeter()
        self.log_recon_2d_loss = AverageMeter()
        self.log_cls_loss = AverageMeter()
        self.log_acc = [AverageMeter() for _ in range(10)]
        self.log_kl_div = AverageMeter()

        model = self.model.to(self.device)
        if self.arg.half:
            self.model, self.optimizer = apex.amp.initialize(
                model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        # self.model = torch.nn.DataParallel(model, device_ids=(0,1,2))

    def load_data(self, A_vector):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_path = f'data/{self.arg.dataset}/{self.arg.datacase}_aligned.npz'
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=data_path,
                split='train',
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=False,
                A=A_vector,
            ),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=init_seed)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                p_interval=[0.95],
                vel=self.arg.use_vel,
                A=A_vector,
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)

    def load_model(self):
        self.model = InfoGCN(
            num_class=self.arg.num_class,
            num_point=self.arg.num_point,
            num_person=self.arg.num_person,
            graph=self.arg.graph,
            in_channels=3,
            num_head=self.arg.n_heads,
            ode_method=self.arg.ode_method,
            k=self.arg.k,
            base_channel=self.arg.base_channel,
            depth=self.arg.depth,
            device=self.device,
            T=self.arg.window_size,
            n_step=self.arg.n_step,
            dilation=self.arg.dilation,
            temporal_pooling=self.arg.temporal_pooling,
            spatial_pooling=self.arg.spatial_pooling,
            SAGC_proj=self.arg.SAGC_proj,
            sigma=self.arg.sigma,
        )
        self.cls_loss = LabelSmoothingCrossEntropy().to(self.device)
        self.recon_loss = masked_recon_loss

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.to(self.device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights, strict=False)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' :
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        with torch.no_grad():
            self.model.eval()
            [self.log_acc[i].reset() for i in range(10)]
            self.log_cls_loss.reset()
            self.log_kl_div.reset()
            self.log_recon_loss.reset()
            self.log_recon_2d_loss.reset()
            self.print_log('Training epoch: {}'.format(epoch + 1))

            label_list = []
            pred_list = []
            tbar = tqdm(self.data_loader['train'], dynamic_ncols=True)

            for x, y, mask, index in tbar:
                cls_loss, recon_loss = torch.tensor(0.), torch.tensor(0.)
                B, C, T, V, M = x.shape;
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                mask = mask.long().to(self.device)
                y_hat, x_hat, _, _= self.model(x)
                label_list.append(y)

                N_cls = y_hat.size(0)//B
                pred_list.append(y_hat.view(N_cls,B,-1,T))
                y = y.view(1,B,1).expand(N_cls, B, y_hat.size(2)).reshape(-1)
                y_hat = rearrange(y_hat, "b i t -> (b t) i")

                N_rec = x_hat.size(0)//B
                x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec)
                for i in range(N_rec):
                    if N_rec == self.arg.n_step:
                        mask_recon[i,:,:,:i+1,:,:] = 0.
                    else:
                        mask_recon[i,:,:,:i,:,:] = 0.
                mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')

                value, predict_label = torch.max(y_hat.data, 1)
                for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                    self.log_acc[i].update((predict_label == y.data)\
                                            .view(N_cls*B,-1)[:,int(math.ceil(T*ratio))-1].float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)
                # self.log_kl_div.update(kl_div.data.item(), B)
                self.log_recon_loss.update(recon_loss.data.item(), B)

                tbar.set_description(
                    f"[Epoch #{epoch}] "\
                    f"ACC_0.5:{self.log_acc[4].avg:.3f}, " \
                )
            pred_list = torch.cat(pred_list, dim=1)
            label = torch.cat(label_list, dim=0)
            alpha, best_correction = find_best_agreement(pred_list, label, T)
        return alpha

    def eval(self, epoch, save_score=False, loader_name=['test'], alpha=1.0):
        self.model.eval()
        [self.log_acc[i].reset() for i in range(10)]
        self.log_cls_loss.reset()
        self.log_kl_div.reset()
        self.log_recon_loss.reset()
        self.log_recon_2d_loss.reset()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        # self.model.set_n_step(self.arg.N)
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            tbar = tqdm(self.data_loader[ln], dynamic_ncols=True)
            for x, y, mask, index in tbar:
                with torch.no_grad():
                    label_list.append(y)
                    B, C, T, V, M = x.shape
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    mask = mask.long().to(self.device)
                    x_gt = x

                    y_hat, x_hat, _, _= self.model(x)
                    N_cls = y_hat.size(0)//B
                    pred_list.append(y_hat.view(N_cls,B,-1,T))
                    y = y.view(1,B,1).expand(N_cls, B, y_hat.size(2)).reshape(-1)
                    y_hat = rearrange(y_hat, "b i t -> (b t) i")
                    cls_loss = self.cls_loss(y_hat, y)
                    N_rec = x_hat.size(0)//B
                    x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                    mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec)

                    _, predict_label = torch.max(y_hat.data, 1)
                    for i in range(N_rec):
                        mask_recon[i,:,:,:i+1,:,:] = 0.
                    mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')
                    step += 1
                for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                    self.log_acc[i].update((predict_label == y.data)\
                                           .view(N_cls,B,-1)[-1,:,int(math.ceil(T*ratio))-1].float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)

                tbar.set_description(
                    f"[Epoch #{epoch}] "\
                    f"ACC_0.5:{self.log_acc[4].avg:.3f}, " \
                )
            pred_list = torch.cat(pred_list, dim=1)
            label_list = torch.cat(label_list, dim=0).to(pred_list.device)
            # alpha = find_best_agreement(pred_list, label_list, T)
            new_array = []
            pred = pred_list.clone().permute(1,3,0,2)
            for t in range(T):
                new_array.append(torch.cat((pred[:,:t,0,:],pred[:,t,:,:]), dim=1))
            y_hat_0 = torch.argmax(pred, dim=3)[:,:,-1]
            corrects_origin = (y_hat_0 == label_list.unsqueeze(1)).sum(0)
            diff, corrects_agree = agreement(new_array, pred_list, label_list, corrects_origin, alpha, T)
            attention = self.model.get_attention()
            pred = pred # B, T, N, Cls

            with open('pred_lst_cat.pkl', 'wb') as f:
                pickle.dump(pred_list.detach().cpu().numpy(), f)
            with open('label_list.pkl', 'wb') as f:
                pickle.dump(label_list.detach().cpu().numpy(), f)
            with open('attention.pkl', 'wb') as f:
                pickle.dump(attention.detach().cpu().numpy(), f)

            eval_dict={f"eval/ACC_{(i+1)/10}":self.log_acc[i].avg for i in range(10)}
            wandb.log(eval_dict)

            eval_dict={f"eval/ACC_{(i+1)/10}":corrects_agree[int(math.ceil(T*(i+1)*0.1))-1]/pred_list.size(1) for i in range(10)}
            wandb.log(eval_dict)


    def start(self):
        self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print_log(f'# Parameters: {count_parameters(self.model)/10**6:.3f}M')
        # alpha = self.train(0)
        # alpha = 0.94
        self.eval(0, alpha=alpha)

def main():
    # parser arguments
    wandb.init(project="agreement")
    parser = get_parser()
    arg = parser.parse_args()
    arg.work_dir = wandb.run.dir
    wandb.config.update(arg)
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
