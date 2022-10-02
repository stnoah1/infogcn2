#!/usr/bin/env python
from __future__ import print_function

import os
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
from einops import rearrange

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

        self.load_optimizer()
        self.load_data()

        self.best_acc = 0
        self.best_acc_epoch = 0
        self.log_recon_loss = AverageMeter()
        self.log_recon_2d_loss = AverageMeter()
        self.log_cls_loss = AverageMeter()
        self.log_acc = [AverageMeter() for _ in range(6)]
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

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_path = f'data/{self.arg.dataset}/{self.arg.datacase}_aligned.npz'
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=data_path,
                split='train',
                window_size=self.arg.window_size,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=False,
                obs=self.arg.obs
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
                window_size=self.arg.window_size,
                p_interval=[0.95],
                vel=self.arg.use_vel,
                obs=self.arg.obs
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
            k=self.arg.k,
            base_channel=self.arg.base_channel,
            device=self.device,
            ratio=self.arg.pred_ratio,
            T=64,
            obs=0.5
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
        self.model.train()
        [self.log_acc[i].reset() for i in range(6)]
        self.log_cls_loss.reset()
        self.log_kl_div.reset()
        self.log_recon_loss.reset()
        self.log_recon_2d_loss.reset()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        lr = self.adjust_learning_rate(epoch)

        tbar = tqdm(self.data_loader['train'], dynamic_ncols=True)

        for x, y, index in tbar:
            B, C, T, V, M = x.shape
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            t = int(T*self.arg.obs)
            x_gt = x.unsqueeze(0).expand(2, B, C, T, V, M).reshape(2*B, C, T, V, M)
            mask = (abs(x).sum(1,keepdim=True).sum(3,keepdim=True) > 0)
            y_hat, x_hat, kl_div = self.model(x)
            y_hat = rearrange(y_hat, "n i t -> (n t) i")
            y = y.unsqueeze(0).unsqueeze(-1).expand(2, B, T).reshape(-1)
            cls_loss = self.cls_loss(y_hat, y)
            if self.arg.dct:
                x_hat_ = rearrange(x_hat, 'b c t v m -> b c v m t')
                x_gt_ = rearrange(x_gt, 'b c t v m -> b c v m t')
                mask_ = rearrange(mask.detach().clone(), 'b c t v m -> b c v m t')
                mask_[:,:,:,:,self.arg.dct_order:] = 0.
                x_hat_dct = dct.dct(x_hat_)
                x_gt_dct = dct.dct(x_gt_)
                recon_loss = self.recon_loss(x_hat_dct, x_gt_dct, mask_)
            else:
                mask = mask.unsqueeze(0).expand(2, B, C, T, V, M).reshape(2*B, C, T, V, M)
                recon_loss = self.recon_loss(x_hat, x_gt, mask)
            recon_eval = self.recon_loss(x_hat[:,:,t:], x_gt[:,:,t:], mask[:,:,t:])
            loss = self.arg.lambda_2 * recon_loss + self.arg.lambda_1 * cls_loss + kl_div

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
            for i, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
                self.log_acc[i].update((predict_label == y.data)\
                                        .view(2*B,T)[:,int(T*ratio)-1].float().mean(), B)
            self.log_cls_loss.update(cls_loss.data.item(), B)
            self.log_kl_div.update(kl_div.data.item(), B)
            self.log_recon_loss.update(recon_loss.data.item(), B)
            self.log_recon_2d_loss.update(recon_eval.data.item(), B)

            tbar.set_description(
                f"[Epoch #{epoch}] "\
                f"ACC_0.5:{self.log_acc[4].avg:.3f}, " \
                f"CLS:{self.log_cls_loss.avg:.3f}, " \
                f"KL:{self.log_kl_div.avg:.3f}, " \
                f"RECON:{self.log_recon_loss.avg:.5f}, " \
                f"RECON2D:{self.log_recon_2d_loss.avg:.5f}, " \
            )
        wandb.log({
            "train/Recon2D_loss":self.log_recon_2d_loss.avg,
            "train/cls_loss":self.log_cls_loss.avg,
            "train/ACC_0.1":self.log_acc[0].avg,
            "train/ACC_0.2":self.log_acc[1].avg,
            "train/ACC_0.3":self.log_acc[2].avg,
            "train/ACC_0.4":self.log_acc[3].avg,
            "train/ACC_0.5":self.log_acc[4].avg,
            "train/ACC_1.0":self.log_acc[5].avg,
        })
        with open(f"results/{self.arg.obs}/x_hat_list_train_{self.arg.base_channel}_{self.arg.base_lr}_{epoch}_{self.arg.dct}.pkl", 'wb') as fp:
            pickle.dump({"x_hat":x_hat, "x":x_gt}, fp)
        # statistics of time consumption and loss
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        [self.log_acc[i].reset() for i in range(6)]
        self.log_cls_loss.reset()
        self.log_kl_div.reset()
        self.log_recon_loss.reset()
        self.log_recon_2d_loss.reset()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            tbar = tqdm(self.data_loader[ln], dynamic_ncols=True)
            for x, y, index in tbar:
                label_list.append(y)
                with torch.no_grad():
                    B, _, T, _, _ = x.shape
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    t = int(T*self.arg.obs)
                    x_gt = x
                    mask = (abs(x).sum(1,keepdim=True).sum(3,keepdim=True) > 0)

                    y_hat, x_hat, kl_div = self.model(x)
                    y_hat = rearrange(y_hat, "n i t -> (n t) i")
                    y = y.unsqueeze(-1).expand(B, T).reshape(-1)
                    cls_loss = self.cls_loss(y_hat, y)
                    if self.arg.dct:
                        x_hat_ = rearrange(x_hat, 'b c t v m -> b c v m t')
                        x_gt_ = rearrange(x_gt, 'b c t v m -> b c v m t')
                        mask_ = rearrange(mask.detach().clone(), 'b c t v m -> b c v m t')
                        mask_[:,:,:,:,self.arg.dct_order:] = 0.
                        x_hat_dct = dct.dct(x_hat_)
                        x_gt_dct = dct.dct(x_gt_)
                        recon_loss = self.recon_loss(x_hat_dct, x_gt_dct, mask_)
                    else:
                        recon_loss = self.recon_loss(x_hat, x_gt, mask)
                    recon_eval = self.recon_loss(x_hat[:,:,t:], x_gt[:,:,t:], mask[:,:,t:])
                    loss = self.arg.lambda_2 * recon_loss + self.arg.lambda_1 * cls_loss
                    score_frag.append(y_hat.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.view(B,T)[:,t].data.cpu().numpy())
                    step += 1
                for i, ratio in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
                    self.log_acc[i].update((predict_label == y.data)\
                                           .view(B,T)[:,int(T*ratio)-1].float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)
                self.log_recon_loss.update(recon_loss.data.item(), B)
                self.log_recon_2d_loss.update(recon_eval.data.item(), B)
                self.log_kl_div.update(kl_div.data.item(), B)

                tbar.set_description(
                    f"[Epoch #{epoch}] "\
                    f"ACC_0.5:{self.log_acc[4].avg:.3f}, " \
                    f"CLS:{self.log_cls_loss.avg:.3f}, " \
                    f"KL:{self.log_kl_div.avg:.3f}, " \
                    f"RECON:{self.log_recon_loss.avg:.5f}, " \
                    f"RECON2D:{self.log_recon_2d_loss.avg:.5f}, " \
                )

            wandb.log({
                "eval/Recon2D_loss":self.log_recon_2d_loss.avg,
                "eval/cls_loss":self.log_cls_loss.avg,
                "eval/ACC_0.1":self.log_acc[0].avg,
                "eval/ACC_0.2":self.log_acc[1].avg,
                "eval/ACC_0.3":self.log_acc[2].avg,
                "eval/ACC_0.4":self.log_acc[3].avg,
                "eval/ACC_0.5":self.log_acc[4].avg,
                "eval/ACC_1.0":self.log_acc[5].avg,
            })
            with open(f"results/{self.arg.obs}/x_hat_list_eval_{self.arg.base_channel}_{self.arg.base_lr}_{epoch}_{self.arg.dct}.pkl", 'wb') as fp:
                pickle.dump({"x_hat":x_hat, "x":x_gt}, fp)

            score = np.concatenate(score_frag)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(cls_loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                with open(f'{self.arg.work_dir}/best_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)/10**6:.3f}M')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1 > self.arg.save_epoch)

                self.train(epoch, save_model=save_model)

                # if epoch > 80:
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)

            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'])
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
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')

def main():
    # parser arguments
    wandb.init()
    parser = get_parser()
    arg = parser.parse_args()
    arg.work_dir = wandb.run.dir
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
