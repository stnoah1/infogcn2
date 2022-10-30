import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torch_dct as dct

from torch import nn

from einops import rearrange

from model.modules import EncodingBlock, SA_GC, TemporalEncoder, GCN
from model.utils import bn_init, import_class, sample_standard_gaussian,\
    cum_mean_pooling, cum_max_pooling, identity, max_pooling
from model.encoder_decoder import Encoder_z0_RNN

from einops import rearrange, repeat
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
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        pred_y = rearrange(pred_y, 't b c e v -> b c t v e').squeeze()

        # b c t v
        return pred_y


class Euler(nn.Module):
    def __init__(self, ode_func, T, n_step, dilation):
        super(Euler, self).__init__()

        self.ode_func = ode_func
        self.n_step = n_step
        self.dilation = dilation
        self.shift_idx = torch.arange(0, T, dtype=int).view(1,1,T,1)

    def forward(self, z):
        """
        # n-step forward with initial point z: (b c t v)
        """
        B, C, T, V = z.size()
        z_shift_lst = [z]
        z_lst = [z]
        shift_tensor = self.shift_idx.expand(B,C,T,V).to(z.device)
        z = rearrange(z, 'b c t v -> (b t) c v')
        for i in range(1, self.n_step+1):
            z = z + self.ode_func(z)

            z_append = rearrange(z, '(b t) c v -> b c t v', t=T)
            z_cls = rearrange(z, '(b t) c v -> b c t v', t=T)
            z_append = torch.gather(z_append, dim=2, index=(shift_tensor-i*self.dilation)%T)
            z_append[:,:,:i*self.dilation,:] = 0
            z_shift_lst.append(z_append)
            z_lst.append(z_cls)
        z_shift_lst = torch.cat(z_shift_lst, dim=0) # n-step*b, c, t, v
        z_cls = torch.cat(z_lst, dim=0) # n-step*b, c, t, v
        return z_shift_lst, z_cls


class RK4(nn.Module):
    def __init__(self, ode_func, T, n_step, dilation):
        super(RK4, self).__init__()
        self.ode_func = ode_func
        self.n_step = n_step
        self.dilation = dilation
        self.shift_idx = torch.arange(0, T, dtype=int).view(1,1,T,1)

    def forward(self, z):
        B, C, T, V = z.size()
        _one_third = 1 / 3
        _two_thirds = 2 / 3
        t = 0
        dt = 1
        z_shift_lst = [z]
        z_lst = [z]
        shift_tensor = self.shift_idx.expand(B,C,T,V).to(z.device)
        z = rearrange(z, 'b c t v -> (b t) c v')
        for i in range(1, self.n_step+1):
            k1 = self.ode_func(z)
            k2 = self.ode_func(z + dt * k1 * _one_third, dt * _one_third)
            k3 = self.ode_func(z + dt * (k2 - k1 * _one_third), dt * _two_thirds)
            k4 = self.ode_func(z + dt * (k1 - k2 + k3), dt)
            z = (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

            z_append = rearrange(z, '(b t) c v -> b c t v', t=T)
            z_cls = rearrange(z, '(b t) c v -> b c t v', t=T)
            z_append = torch.gather(z_append, dim=2, index=(shift_tensor-i*self.dilation)%T)
            z_append[:,:,:i*self.dilation,:] = 0
            z_shift_lst.append(z_append)
            z_lst.append(z_cls)
        z_hat = torch.cat(z_shift_lst, dim=0)
        z_cls = torch.cat(z_lst, dim=0) # n-step*b, c, t, v
        return z_hat, z_cls


class SDE(nn.Module):
    def __init__(self, mu, sigma, T, n_step, dilation=1):
        super(SDE, self).__init__()

        self.mu = mu
        self.sigma = sigma
        self.n_step = n_step
        self.dilation = dilation
        self.shift_idx = torch.arange(0, T, dtype=int).view(1,1,T,1)
        self.n_sample = 1

    def forward(self, z):
        """
        # n-step forward with initial point z: (b c t v)
        """
        B, C, T, V = z.size()
        z = repeat(z, 'b c t v -> (b l) c t v', l=self.n_sample)
        z_lst = [z]
        z_shift_lst = [z]
        z = rearrange(z, '(b l) c t v -> (b t l) c v', l=self.n_sample)
        shift_tensor = self.shift_idx.expand(B*self.n_sample,C,T,V).to(z.device)
        dt = 1
        for i in range(1, self.n_step+1):
            if self.training:
                z = z \
                    + self.mu(z) * dt \
                    + self.sigma(z) * self.dW(z) \
                    + 0.5 * self.sigma(z)**2 * z * (self.dW(z)**2 - dt)
            else:
                z = z \
                    + self.mu(z) * dt

            z_append = rearrange(z, '(b t l) c v -> (b l) c t v', t=T, l=self.n_sample)
            z_cls = rearrange(z, '(b t l) c v -> (b l) c t v', t=T, l=self.n_sample)
            z_append = torch.gather(z_append, dim=2, index=(shift_tensor-i*self.dilation)%T)
            z_append[:,:,:i*self.dilation,:] = 0
            z_shift_lst.append(z_append)
            z_lst.append(z_cls)
        z_shift_lst = torch.cat(z_shift_lst, dim=0) # n-step*b*m, c, t, v
        z_cls = torch.cat(z_lst, dim=0) # n-step*b*m, c, t, v
        return z_shift_lst, z_cls

    def dW(self, z, dt=1):
        if not self.training and (self.n_sample != 1):
            B, C, V = z.shape
            noise = torch.normal(torch.zeros((B,C,V), device=z.device), \
                            torch.ones((B,C,V), device=z.device)*math.sqrt(dt))
            return noise
        else:
            return torch.normal(torch.zeros_like(z, device=z.device), \
                            torch.ones_like(z, device=z.device)*math.sqrt(dt))

    def set_n_sample(self, n):
        self.n_sample = n


class ODEFunc(nn.Module):
    def __init__(self, dim, A, T=64):
        super(ODEFunc, self).__init__()
        self.layers = []
        self.A = A
        self.T = T
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.conv3 = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.temporal_pe = self.init_pe(3*T+1, dim)
        self.index = torch.arange(0, 3*T, 3).unsqueeze(-1).expand(T, dim)

    def forward(self, x, t=0, backwards=False):
        x = self.add_pe(x, t) # TODO T dimension wise.
        x = torch.einsum('vu,ncu->ncv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.einsum('vu,ncu->ncv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = torch.einsum('vu,ncu->ncv', self.A.to(x.device).to(x.dtype), x)
        # x = self.conv3(x)
        # x = self.relu(x)
        x = self.proj(x)

        return x

    def init_pe(self, length, d_model):
        with torch.no_grad():
            import math
            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                                -(math.log(10000.0) / d_model))) # length, model
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.view(length, d_model)
        return pe

    def add_pe(self, x, t):
        B, C, V = x.size()
        T = self.T
        # pe = self.temporal_pe.view(1, T, -1, 1, 1).expand(B//T, T, C, 1, V)
        index = self.index.to(x.device) + int(t*3)
        pe = torch.gather(self.temporal_pe.to(x.device), dim=0, index=index)
        pe = repeat(pe, 't c -> b t c v', v=V, b=B//T)
        x = rearrange(x, '(b t) c v -> b t c v', t=T) # .view(-1,T,C,1,V)
        x = x + pe
        x = rearrange(x, 'b t c v -> (b t) c v')
        return x

class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, ode_method='rk4',
                 graph=None, in_channels=3, num_head=3, k=0, base_channel=64, depth=4, device='cuda',
                 dct=True, T=64, n_step=1, dilation=1, temporal_pooling="None", spatial_pooling="None", SAGC_proj=True, n_sample=1, sigma=None):
        super(InfoGCN, self).__init__()

        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        self.Graph = import_class(graph)()
        A = np.stack([self.Graph.A_norm] * num_head, axis=0)
        self.dct = dct
        self.T = T
        method = ode_method
        self.arange = torch.arange(T).view(1,1,T)+1

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.A_vector = self.get_A(k)
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        # self.data_bn = nn.BatchNorm1d(base_channel)
        # bn_init(self.data_bn, 1)
        self.temporal_encoder = TemporalEncoder(
            seq_len=T,
            dim=base_channel,
            depth=depth,
            heads=4,
            mlp_dim=base_channel*2,
            dim_head=base_channel//4,
            device=device,
            A=A,
            num_point=num_point,
            SAGC_proj=SAGC_proj
        )
        self.n_sample = n_sample

        # self.diffeq_solver = DiffeqSolver(ode_func, method='rk4')
        if method == "sde":
            mu = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=T).to(device)
            sigma = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=64).to(device) if sigma is None else lambda x: sigma
            self.n_step_ode = SDE(mu, sigma, T=T, n_step=n_step, dilation=dilation)
        elif method == "rk4":
            ode_func = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=T).to(device)
            self.n_step_ode = RK4(ode_func, T=T, n_step=n_step, dilation=dilation)
        elif method == "euler":
            ode_func = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=T).to(device)
            self.n_step_ode = Euler(ode_func, T=T, n_step=n_step, dilation=dilation)

        self.recon_decoder = nn.Sequential(
            GCN(base_channel, base_channel, A),
            GCN(base_channel, base_channel, A),
            nn.Conv2d(base_channel, 3, 1),
        )

        self.cls_decoder = nn.Sequential(
            GCN(base_channel, base_channel, A),
            GCN(base_channel, base_channel, A),
        )

        self.classifier = nn.Conv1d(base_channel, num_class, 1)

        if temporal_pooling == "max":
            self.temporal_pooling = cum_max_pooling
        elif temporal_pooling == "mean":
            self.temporal_pooling = cum_mean_pooling
        elif temporal_pooling == "None":
            self.temporal_pooling = identity

        if spatial_pooling == "max":
            self.spatial_pooling = max_pooling
        elif spatial_pooling == "mean":
            self.spatial_pooling = torch.mean

        # self.zero_embed = nn.Parameter(torch.randn(2*base_channel, 25))


    def KL_div(self, fp_mu, fp_std, kl_coef=1):
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        loss = kldiv_z0.mean()
        return loss

    def get_A(self, k):
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def set_n_sample(self, n_sample):
        self.n_sample = n_sample
        self.n_step_ode.set_n_sample(n_sample)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)
        # x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

        # embedding
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :self.num_point]

        # encoding
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N)
        z = self.temporal_encoder(x)
        # kl_div = self.KL_div(z_mu, z_std)
        # z = sample_standard_gaussian(z_mu, z_std)

        # z = z_mu

        # extrapolation
        z_shift_lst, z_cls = self.n_step_ode(z)

        # reconstruction
        x_hat = self.recon_decoder(z_shift_lst)
        x_hat = rearrange(x_hat, '(n m l) c t v -> n l c t v m', m=M, l=self.n_sample).mean(1)

        # classification
        z_cls = self.cls_decoder(z_cls)
        z_cls = rearrange(z_cls, '(n m l) c t v -> (n l) m c t v', m=M, l=self.n_sample).mean(1) # N, 2*D, T
        z_cls = self.spatial_pooling(z_cls,dim=-1)
        z_cls = self.temporal_pooling(z_cls, self.arange.to(z_cls.device), dim=-1)

        y = self.classifier(z_cls) # N, num_cls, T
        y = rearrange(y, '(n l) c t -> n l c t', l=self.n_sample).mean(1)
        return y, x_hat, torch.tensor(0.)
