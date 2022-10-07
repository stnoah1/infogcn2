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


class NStepODE(nn.Module):
    def __init__(self, ode_func, method, T, n_step):
        super(NStepODE, self).__init__()

        self.ode_func = ode_func
        self.n_step = n_step
        self.shift_idx = torch.arange(0, T, dtype=int).view(1,1,T,1)

    def forward(self, z):
        """
        # n-step forward with initial point z: (b c t v)
        """
        B, C, T, V = z.size()
        z_pred = []
        shift_tensor = self.shift_idx.expand(B,C,T,V).to(z.device)
        z = rearrange(z, 'b c t v -> (b t) c v').view(B*T, C, 1, V)
        for i in range(self.n_step):
            z = z + self.ode_func(z)

            shift_tensor = (shift_tensor-1)%T
            z_append = rearrange(z, '(b t) c i v -> b c t i v', t=T).squeeze()
            z_append = torch.gather(z_append, dim=2, index=shift_tensor)
            z_append[:,:,:i+1,:] = 0
            z_pred.append(z_append)
        z_pred = torch.cat(z_pred, dim=0) # n-step*b, c, t, v
        return z_pred


class ODEFunc(nn.Module):
    def __init__(self, dim, A, T=64):
        super(ODEFunc, self).__init__()
        self.layers = []
        self.A = A
        self.T = T
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.temporal_pe = self.init_pe(T, dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, dim, T*2, 1))
        # self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x, backwards=False):
        x = self.add_pe(x) # TODO T dimension wise.
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

    def add_pe(self, x):
        B, C, _, V = x.size()
        T = self.T
        # pe = self.temporal_pe.view(1, T, -1, 1, 1).expand(B//T, T, C, 1, V)
        pe = repeat(self.temporal_pe, 't c -> b t c 1 v', v=V, b=B//T).to(x.device)
        x = rearrange(x, '(b t) c i v -> b t c i v', t=T) # .view(-1,T,C,1,V)
        x = x + pe
        x = rearrange(x, 'b t c i v -> (b t) c i v')
        return x

class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, ode_solver_method='rk4',
                 graph=None, in_channels=3, num_head=3, k=0, base_channel=64, depth=4, device='cuda',
                 dct=True, T=64, n_step=1):
        super(InfoGCN, self).__init__()

        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        self.Graph = import_class(graph)()
        A = np.stack([self.Graph.A_norm] * num_head, axis=0)
        self.dct = dct
        self.T = T

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
            num_point=num_point
        )

        # self.diffeq_solver = DiffeqSolver(ode_func, method='rk4')
        ode_func = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=64).to(device)
        self.n_step_ode = NStepODE(ode_func, method='rk4', T=T, n_step=n_step)

        self.recon_decoder = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            SA_GC(base_channel, base_channel, A),
            nn.Conv2d(base_channel, 3, 1),
        )

        self.cls_decoder = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            SA_GC(base_channel, base_channel, A),
            nn.ReLU(),
        )


        self.classifier = nn.Conv1d(base_channel, num_class, 1)

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

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)
        x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

        # embedding
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :self.num_point]

        # encoding
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N)
        z = self.temporal_encoder(x)

        # extrapolation
        if self.training:
            z_pred = self.n_step_ode(z)
        else:
            z_pred = z
        # reconstruction
        x_hat = self.recon_decoder(z_pred)
        x_hat = rearrange(x_hat, '(n m) c t v -> n c t v m', m=M)

        # classification
        z_cls = self.cls_decoder(z)
        z_cls = rearrange(z_cls, '(n m) c t v -> n m c t v', m=M).mean(-1).mean(1) # N, 2*D, T
        y = self.classifier(z_cls) # N, num_cls, T
        kl_div = torch.tensor(0.0)
        return y, x_hat, kl_div
