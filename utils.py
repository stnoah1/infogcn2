import argparse
import random

import sys
import traceback
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

# def import_class(name):
    # components = name.split('.')
    # mod = __import__(components[0])
    # for comp in components[1:]:
        # mod = getattr(mod, comp)
    # return mod

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_masked_input_and_labels(inp, mask_value=1, mask_p=0.15, mask_random_p=0.1, mask_remain_p=0.1, mask_random_s=1):
    # BERT masking
    inp_mask = (torch.rand(*inp.shape[:2]) < mask_p).to(inp.device)

    # Prepare input
    inp_masked = inp.clone().float()

    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = (inp_mask & (torch.rand(*inp.shape[:2]) < 1 - mask_remain_p).to(inp.device))
    inp_masked[inp_mask_2mask] = mask_value # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (torch.rand(*inp.shape[:2]) < mask_random_p / (1 - mask_remain_p)).to(inp.device)

    inp_masked[inp_mask_2random] = (2 * mask_random_s * torch.rand(inp_mask_2random.sum().item(), inp.shape[2]) - mask_random_s).to(inp.device)

    # y_labels would be same as encoded_texts i.e input tokens
    gt = inp.clone()
    return inp_masked, gt

def random_rot_mat(bs, uniform_dist):
    rot_mat = torch.zeros(bs, 3, 3)
    random_values = uniform_dist.rsample((bs,))
    rot_mat[:, 0, 0] = torch.cos(random_values)
    rot_mat[:, 0, 1] = -torch.sin(random_values)
    rot_mat[:, 1, 0] = torch.sin(random_values)
    rot_mat[:, 1, 1] = torch.cos(random_values)
    rot_mat[:, 2, 2] = 1
    return rot_mat

def repeat_rot_mat(rot_mat, num):
    batch = rot_mat.shape[0]
    res = torch.zeros([batch, 3*num, 3*num]).to(rot_mat.device)
    for i in range(num):
        res[:, 3*i:3*(i+1), 3*i:3*(i+1)] = rot_mat
    return res

def align_skeleton(data):
    N, C, T, V, M = data.shape
    trans_data = np.zeros_like(data)
    for i in tqdm(range(N)):
        for p in range(M):
            sample = data[i][..., p]
            # if np.all((sample[:,0,:] == 0)):
                # continue
            d = sample[:,0,1:2]
            v1 = sample[:,0,1]-sample[:,0,0]
            if np.linalg.norm(v1) <= 0.0:
                continue
            v1 = v1/np.linalg.norm(v1)
            v2_ = sample[:,0,12]-sample[:,0,16]
            proj_v2_v1 = np.dot(v1.T,v2_)*v1/np.linalg.norm(v1)
            v2 = v2_-np.squeeze(proj_v2_v1)
            v2 = v2/(np.linalg.norm(v2))
            v3 = np.cross(v2,v1)/(np.linalg.norm(np.cross(v2,v1)))
            v1 = np.reshape(v1,(3,1))
            v2 = np.reshape(v2,(3,1))
            v3 = np.reshape(v3,(3,1))

            R = np.hstack([v2,v3,v1])
            for t in range(T):
                trans_sample = (np.linalg.inv(R))@(sample[:,t,:]) # -d
                trans_data[i, :, t, :, p] = trans_sample
    return trans_data

def create_aligned_dataset(file_list=['data/ntu/NTU60_CS.npz', 'data/ntu/NTU60_CV.npz']):
    for file in file_list:
        org_data = np.load(file)
        splits = ['x_train', 'x_test']
        aligned_set = {}
        for split in splits:
            data = org_data[split]
            N, T, _ = data.shape
            data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
            aligned_data = align_skeleton(data)
            aligned_data = aligned_data.transpose(0, 2, 4, 3, 1).reshape(N, T, -1)
            aligned_set[split] = aligned_data

        np.savez(file.replace('.npz', '_aligned.npz'),
                 x_train=aligned_set['x_train'],
                 y_train=org_data['y_train'],
                 x_test=aligned_set['x_test'],
                 y_test=org_data['y_test'])



def get_motion(data, data_format=['x'], use_nonzero_mask=False, rot=False, jittering=False, random_dist=None):
    N, C, T, V, M = data.size()
    data = data.permute(0, 4, 2, 3, 1).contiguous().view(N*M, T, V, C)

    # get motion features
    x = data - data[:,:,0:1,:] # localize
    if 'v' in data_format:
        v = x[:,1:,:,:] - x[:,:-1,:,:]
        v = torch.cat([torch.zeros(N*M, 1, V, C).to(v.device), v], dim=1)
    if 'a' in data_format:
        a = v[:,1:,:,:] - v[:,:-1,:,:]
        a = torch.cat([torch.zeros(N*M, 1, V, C).to(a.device), a], dim=1)

    # reshape x,v for PORT
    x = x.view(N*M*T, V, C)
    if 'v' in data_format:
        v = v.view(N*M*T, V, C)
    if 'a' in data_format:
        a = a.view(N*M*T, V, C)

    # apply nonzero mask
    if use_nonzero_mask:
        nonzero_mask = x.view(N*M*T, -1).count_nonzero(dim=-1) !=0
        x = x[nonzero_mask]
        if 'v' in data_format:
            v = v[nonzero_mask]
        if 'a' in data_format:
            a = a[nonzero_mask]

    # optionally rotate
    if rot:
        rot_mat = random_rot_mat(x.shape[0], random_dist).to(x.device)
        x = x.transpose(1, 2) # (NMT, C, V)
        x = torch.bmm(rot_mat, x) # rotate
        x = x.transpose(1, 2) #(NMT, V, C)

        if 'v' in data_format:
            v = v.transpose(1, 2) # (NMT, C, V)
            v = torch.bmm(rot_mat, v) # rotate
            v = v.transpose(1, 2) #(NMT, V, C)

        if 'a' in data_format:
            a = a.transpose(1, 2) # (NMT, C, V)
            a = torch.bmm(rot_mat, a) # rotate
            a = a.transpose(1, 2) #(NMT, V, C)

    if jittering:
        jit = (torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) - 0.5) / 10
        x += jit

    output = {'x':x}
    if 'v' in data_format:
        output['v'] = v
    if 'a' in data_format:
        output['a'] = a

    return output

def get_attn(x, mask= None, similarity='scaled_dot'):
    if similarity == 'scaled_dot':
        sqrt_dim = np.sqrt(x.shape[-1])
        score = torch.bmm(x, x.transpose(1, 2)) / sqrt_dim
    elif similarity == 'euclidean':
        score = torch.cdist(x, x)

    if mask is not None:
        score.masked_fill_(mask.view(score.size()), -float('Inf'))

    attn = F.softmax(score, -1)
    embd = torch.bmm(attn, x)
    return embd, attn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == "__main__":
    create_aligned_dataset()
