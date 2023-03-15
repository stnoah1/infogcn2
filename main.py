#!/usr/bin/env python from __future__ import print_function

import os
import math
import time
import glob
import wandb
import pickle
import random
import resource
import time

from collections import OrderedDict

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_dct as dct
import numpy as np

from tqdm import tqdm

from args import get_parser
from loss import LabelSmoothingCrossEntropy, masked_recon_loss
from model.sode import SODE
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
        self.log_auc = AverageMeter()
        self.log_recon_2d_loss = AverageMeter()
        self.log_cls_loss = AverageMeter()
        self.log_kl_div = AverageMeter()
        self.log_acc = [AverageMeter() for _ in range(10)]
        self.log_feature_loss = AverageMeter()

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
                window_size=self.arg.window_size,
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
                window_size=self.arg.window_size,
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)

    def load_model(self):
        self.model = SODE(
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
            SAGC_proj=self.arg.SAGC_proj,
            backbone=self.arg.backbone,
            num_cls=self.arg.num_cls,
        )
        self.cls_loss = LabelSmoothingCrossEntropy(T=self.arg.window_size).to(self.device)
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
        self.model.train()
        [self.log_acc[i].reset() for i in range(10)]
        self.log_cls_loss.reset()
        self.log_auc.reset()
        self.log_feature_loss.reset()
        self.log_recon_loss.reset()
        self.log_recon_2d_loss.reset()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        lr = self.adjust_learning_rate(epoch)
        idx10 = np.array([int(math.ceil(self.arg.window_size*ratio*0.1))-1 for ratio in range(10)])
        tbar = tqdm(self.data_loader['train'], dynamic_ncols=True)

        for x, y, mask, index in tbar:
            cls_loss, recon_loss, feature_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            B, C, T, V, M = x.shape;
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            mask = mask.long().to(self.device)
            y_hat, x_hat, z_0, z_hat, kl_div = self.model(x)
            N_cls = y_hat.size(0)//B

            if self.arg.lambda_1:
                y = y.view(1,B,1).expand(N_cls, B, y_hat.size(2))
                y_hat_ = rearrange(y_hat, "b i t -> (b t) i")

                cls_loss = self.arg.lambda_1 * self.cls_loss(y_hat_, y.reshape(-1))

            if self.arg.lambda_2:
                N_rec = x_hat.size(0)//B
                x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec).clone()
                for i in range(N_rec):
                    if N_rec == self.arg.n_step:
                        mask_recon[i,:,:,:i+1,:,:] = 0.
                    else:
                        mask_recon[i,:,:,:i,:,:] = 0.
                mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')
                recon_loss = self.arg.lambda_2 * self.recon_loss(x_hat, x_gt, mask_recon)

            if self.arg.lambda_3:
                N_step = self.arg.n_step
                B_,C,T,V = z_0.shape
                z_0 = repeat(z_0, 'b c t v-> n b c t v', n=N_step)
                z_hat = z_hat.view(N_step, B_, C, T, V)
                mask_feature = (z_hat != 0.)
                feature_loss = self.arg.lambda_3 * self.recon_loss(z_hat, z_0, mask_feature)# F.mse_loss(z_0, z_hat)#

            if self.arg.lambda_4:
                kl_div = self.arg.lambda_4 * kl_div

            loss = cls_loss + recon_loss + feature_loss + kl_div

            # backward
            self.optimizer.zero_grad()
            if self.arg.half:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            value, predict_label = torch.max(y_hat.data, 1)
            for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                self.log_acc[i].update((predict_label == y.data)\
                                        .view(N_cls*B,-1)[:,int(math.ceil(T*ratio))-1].float().mean(), B)
            self.log_cls_loss.update(cls_loss.data.item(), B)
            self.log_feature_loss.update(feature_loss.data.item(), B)
            self.log_recon_loss.update(recon_loss.data.item(), B)
            self.log_kl_div.update(kl_div.data.item(), B)

            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            tbar.set_description(
                f"[Epoch #{epoch}] "\
                f"AUC:{AUC:.3f}, " \
                f"CLS:{self.log_cls_loss.avg:.3f}, " \
                f"FT:{self.log_feature_loss.avg:.3f}, " \
                f"RECON:{self.log_recon_loss.avg:.5f}, " \
            )
        AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
        train_dict = {
            "train/Recon2D_loss":self.log_recon_loss.avg,
            "train/cls_loss":self.log_cls_loss.avg,
            "train/feature_loss":self.log_feature_loss.avg,
            "train/kl_div":self.log_kl_div.avg,
            "train/AUC":AUC,
        }
        train_dict.update({f"train/ACC_{(i+1)/10}":self.log_acc[i].avg for i in range(10)})
        wandb.log(train_dict)
        # statistics of time consumption and loss
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        [self.log_acc[i].reset() for i in range(10)]
        self.log_cls_loss.reset()
        self.log_feature_loss.reset()
        self.log_recon_loss.reset()
        self.log_recon_2d_loss.reset()
        self.log_kl_div.reset()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            time_lst = []
            tbar = tqdm(self.data_loader[ln], dynamic_ncols=True)
            idx10 = np.array([int(math.ceil(self.arg.window_size*0.1*i))-1 for i in range(1,11)])
            for x, y, mask, index in tbar:
                label_list.append(y)
                with torch.no_grad():
                    B, C, T, V, M = x.shape
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    y_ = y.clone()
                    mask = mask.long().to(self.device)
                    x_gt = x
                    y_hat, x_hat, z_0, z_hat, kl_div = self.model(x)
                    N_cls = y_hat.size(0)//B
                    y = y.view(1,B,1).expand(N_cls, B, y_hat.size(2)).reshape(-1)
                    y_hat = rearrange(y_hat, "b i t -> (b t) i")
                    cls_loss = self.cls_loss(y_hat, y)
                    N_rec = x_hat.size(0)//B
                    x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                    mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec)
                    for i in range(N_rec):
                        if N_rec == self.arg.n_step:
                            mask_recon[i,:,:,:i+1,:,:] = 0.
                        else:
                            mask_recon[i,:,:,:i,:,:] = 0.
                    mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')
                    recon_loss = self.recon_loss(x_hat, x_gt, mask_recon)

                    N_step = self.arg.n_step
                    B_,C,T,V = z_0.shape
                    z_0 = repeat(z_0, 'b c t v-> n b c t v', n=N_step)
                    z_hat = z_hat.view(N_step, B_, C, T, V)
                    mask_feature = (z_hat != 0.)
                    feature_loss = self.recon_loss(z_0, z_hat, mask_feature)

                    loss = self.arg.lambda_2 * recon_loss + self.arg.lambda_1 * cls_loss
                    score_frag.append(y_hat.view(B,T,-1)[:,:,:].data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    step += 1
                for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                    self.log_acc[i].update((predict_label == y.data)\
                                            .view(N_cls*B,-1)[:,int(math.ceil(T*ratio))-1].float().mean(), B)
                self.log_auc.update((predict_label == y.data)\
                                    .view(N_cls,B,-1)[-1,:,:].float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)
                self.log_recon_loss.update(recon_loss.data.item(), B)
                self.log_feature_loss.update(feature_loss.data.item(), B)
                self.log_kl_div.update(kl_div.data.item(), B)

                AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
                tbar.set_description(
                    f"[Epoch #{epoch}] "\
                    f"AUC:{AUC:.3f}, " \
                    f"CLS:{self.log_cls_loss.avg:.3f}, " \
                    f"FT:{self.log_feature_loss.avg:.3f}, " \
                    f"RECON:{self.log_recon_loss.avg:.5f}, " \
                )
            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            eval_dict = {
                "eval/Recon2D_loss":self.log_recon_loss.avg,
                "eval/cls_loss":self.log_cls_loss.avg,
                "eval/feature_loss":self.log_feature_loss.avg,
                "eval/kl_div":self.log_kl_div.avg,
                "eval/AUC":AUC,
            }
            eval_dict.update({f"eval/ACC_{(i+1)/10}":self.log_acc[i].avg for i in range(10)})
            wandb.log(eval_dict)

            score = np.concatenate(score_frag)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))

            if save_score and ((epoch+1) >= (self.arg.num_epoch-10)):
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)/10**6:.3f}M')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1 > self.arg.save_epoch)
                self.train(epoch, save_model=save_model)
                if epoch > self.arg.save_epoch:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')

def main():
    # parser arguments
    parser = get_parser()
    arg = parser.parse_args()
    wandb.init(project=arg.project)
    arg.work_dir = wandb.run.dir
    wandb.config.update(arg)
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
