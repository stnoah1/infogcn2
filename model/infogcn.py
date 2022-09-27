import numpy as np
import random
import torch
import torch.nn.functional as F
import torch_dct as dct

from torch import nn

from einops import rearrange

from model.modules import EncodingBlock, SA_GC, TemporalEncoder
from model.utils import bn_init, import_class
from model.encoder_decoder import Encoder_z0_RNN

from einops import rearrange
from torchdiffeq import odeint as odeint


from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torchdiffeq import odeint as odeint


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        """
        # Decode the trajectory through ODE Solver
        """
        # max_order should be set tonumber of obs
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        pred_y = rearrange(pred_y, 't b c e v -> b c t v e').squeeze()

        # b c t v
        return pred_y


class ODEFunc(nn.Module):
    def __init__(self, dim, A, T=64):
        super(ODEFunc, self).__init__()
        self.layers = []
        self.A = A
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.temporal_pe = self.init_pe(T, dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, dim, T*2, 1))
        # self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, t, x, backwards=False):
        # TODO:refactroing
        t = int(t.item())
        x = x + self.temporal_pe[:,:,t:t+1,:]
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv1(x)
        x = self.tanh(x)
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = self.proj(x)

        return x


    def init_pe(self, length, d_model):
        import math
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.transpose(-2,-1).view(1, d_model, length, 1)
        return pe.cuda() # replace cuda to device

class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, ode_solver_method='rk4',
                 graph=None, in_channels=3, num_head=3, k=0, base_channel=64, device='cuda',
                 dct=True, ratio=0.9, T=64, obs=0.5):
        super(InfoGCN, self).__init__()

        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        self.Graph = import_class(graph)()
        A = np.stack([self.Graph.A_norm] * num_head, axis=0)
        self.dct = dct
        self.T = T
        self.obs = obs

        self.num_class = num_class
        self.num_point = num_point
        self.A_vector = self.get_A(k)
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        bn_init(self.data_bn, 1)
        ode_func = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=64).to(device)
        self.temporal_encoder = TemporalEncoder(64, base_channel, base_channel, device=device)

        self.diffeq_solver = DiffeqSolver(ode_func, method='rk4')

        self.spatial_encoder = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            # SA_GC(base_channel, base_channel, A),
            # SA_GC(base_channel, 2*base_channel, A),
            nn.Conv2d(base_channel, base_channel, 1),
        )

        self.spatial_encoder2 = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            # SA_GC(base_channel, base_channel, A),
            # SA_GC(base_channel, 2*base_channel, A),
            nn.Conv2d(base_channel, base_channel, 1),
        )

        self.recon_decoder = nn.Sequential(
            SA_GC(base_channel, 2*base_channel, A),
            SA_GC(2*base_channel, base_channel, A),
            nn.Conv2d(base_channel, 3, 1),
        )

        self.cls_decoder =  nn.Sequential(
            EncodingBlock(base_channel, base_channel, A),
            EncodingBlock(base_channel, base_channel, A),
            EncodingBlock(base_channel, base_channel, A),
            EncodingBlock(base_channel, base_channel*2, A, stride=2),
            EncodingBlock(base_channel*2, base_channel*2, A),
            EncodingBlock(base_channel*2, base_channel*2, A),
            EncodingBlock(base_channel*2, base_channel*4, A, stride=2),
            EncodingBlock(base_channel*4, base_channel*4, A),
            EncodingBlock(base_channel*4, base_channel*4, A),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4*base_channel, num_class)
        )

        self.zero_embed = nn.Parameter(torch.randn(base_channel, 25))



    def KL_div(self, fp_mu, fp_std, kl_coef=1):
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        loss = kldiv_z0.mean()
        return loss

    def extrapolate(self, z):
        '''
        z : (n m) c t v
        '''
        obs_range = int(self.T * self.obs)
        t = torch.linspace(0, self.T, self.T+1).to(z.device)
        if self.training:
            z_hat = []
            for i in range(obs_range):
                z_i = self.diffeq_solver(z[:, :, i:i+1, :], t[:2])
                z_hat.append(z_i[:,:,-1:,:])
            z_hat = torch.cat(z_hat, dim=2)
        else:
            z_hat = z[:, :, :obs_range, :]

        z_0 = z[:, :, obs_range-1:obs_range, :]
        z_pred = self.diffeq_solver(z_0, t[:self.T-obs_range+1])
        z_hat = torch.cat((z_hat, z_pred[:,:,1:,:]), dim=2)

        return z_hat

    def get_A(self, k):
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x, t, mask):
        N, C, T, V, M = x.size()
        z = self.encode(x, t)
        z = self.extrapolate(z)
        mask_ = rearrange(mask.detach().clone(), 'n c t v m -> (n m) c t v', m=M, n=N)
        z = torch.where(mask_.expand_as(z), z, self.zero_embed.view(-1,64,1,25).expand_as(z).to(z.dtype))
        x_hat = self.reconstruct(z, N)
        y = self.classify(z, N, M) #if reconstruction_only else torch.zeros(N,60).to(x.device).to(x.dtype)
        kl_div = torch.tensor(0.0)
        return y, x_hat, kl_div

    def finetune(self, x, t, mask, training=True):
        with torch.no_grad():
            N, C, T, V, M = x.size()
            z = self.encode(x, t)
            z = self.diffeq_solver(z, t, training=False)
            mask_ = rearrange(mask.detach().clone(), 'n c t v m -> (n m) c t v', m=M, n=N)
            z = torch.where(mask_.expand_as(z), z, self.zero_embed.view(-1,64,1,25).expand_as(z).to(z.dtype))
            x_recon = self.reconstruct(z, N)
        z = self.encode2(x_recon, t)
        y = self.classify(z, N, M)
        kl_div = torch.tensor(0.0)
        return y, x_recon, kl_div

    def encode(self, x, t):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)
        x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

        # embedding
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :self.num_point]

        # encoding
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N)
        z = self.spatial_encoder(x)
        return z

    def encode2(self, x, t):
        with torch.no_grad():
            N, C, T, V, M = x.size()
            x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)
            x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

            # embedding
            x = self.to_joint_embedding(x)
            x = x + self.pos_embedding[:, :self.num_point]

            # encoding
            x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N)
        z = self.spatial_encoder2(x)
        return z

    def reconstruct(self, z, N):
        _, C, T, V = z.size()
        x_hat = self.recon_decoder(z)
        x_hat = rearrange(x_hat, '(n m) c t v -> n c t v m', n=N)
        return x_hat

    def classify(self, z, N, M):
        _, C, T, V = z.size()
        y = self.cls_decoder(z)
        y = y.view(N, M, y.size(1), -1).mean(3).mean(1)
        y = self.classifier(y)
        return y

