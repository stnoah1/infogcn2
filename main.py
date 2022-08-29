#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import glob
import pickle
import random
import traceback
import resource

from collections import OrderedDict

import apex
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from args import get_parser
from loss import LabelSmoothingCrossEntropy
from model.infogcn import InfoGCN
from utils import AverageMeter

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

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


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
        self.log_cls_loss = AverageMeter()
        self.log_acc = AverageMeter()



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
                window_size=64,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=False,
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
                window_size=64,
                p_interval=[0.95],
                vel=self.arg.use_vel
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
            device=self.device,
        )
        self.cls_loss = LabelSmoothingCrossEntropy().to(self.device)
        self.recon_loss = nn.MSELoss().to(self.device)

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
                self.model.load_state_dict(weights)
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
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
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
        self.log_acc.reset()
        self.log_cls_loss.reset()
        self.log_recon_loss.reset()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch)

        tbar = tqdm(self.data_loader['train'], dynamic_ncols=True)

        for x, y, index in tbar:
            B, _, T, _, _ = x.shape
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            t = torch.linspace(0, T - 1, T).to(self.device)

            # forward
            y_hat, x_hat = self.model(x[:, :, :int(self.arg.obs*T), ...], t)

            cls_loss = self.cls_loss(y_hat, y)
            recon_loss = self.recon_loss(x_hat, x)
            loss = cls_loss + self.arg.lambda_1 * recon_loss

            # backward
            self.optimizer.zero_grad()
            if self.arg.half:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            value, predict_label = torch.max(y_hat.data, 1)

            self.log_acc.update((predict_label == y.data).float().mean(), B)
            self.log_cls_loss.update(cls_loss.data.item(), B)
            self.log_recon_loss.update(recon_loss.data.item(), B)

            tbar.set_description(
                f"[Epoch #{epoch}]"\
                f"ACC:{self.log_acc.avg:.3f}, " \
                f"CLS_loss:{self.log_cls_loss.avg:.3f}, " \
                f"RECON_loss:{self.log_recon_loss.avg:.3f}, " \
            )

        # statistics of time consumption and loss
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        self.log_acc.reset()
        self.log_cls_loss.reset()
        self.log_recon_loss.reset()
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
                    B, _, T, _, _ = x.shapewq
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    t = torch.linspace(0, T - 1, T).to(self.device)

                    y_hat, x_hat = self.model(x[:, :, :int(self.arg.obs*T), ...], t)
                    cls_loss = self.cls_loss(y_hat, y)
                    recon_loss = self.recon_loss(x_hat, x)
                    loss = cls_loss + self.arg.lambda_1 * recon_loss
                    score_frag.append(y_hat.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                self.log_acc.update((predict_label == y.data).float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)
                self.log_recon_loss.update(recon_loss.data.item(), B)

                tbar.set_description(
                    f"[Epoch #{epoch}]"\
                    f"ACC:{self.log_acc.avg:.3f}, " \
                    f"CLS_loss:{self.log_cls_loss.avg:.3f}, " \
                    f"RECON_loss:{self.log_recon_loss.avg:.3f}, " \
                )


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
                save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)

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
    parser = get_parser()
    arg = parser.parse_args()
    arg.work_dir = f"results/{arg.dataset}_{arg.datacase}"
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
