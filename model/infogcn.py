import numpy as np
import torch
import torch.nn.functional as F
import torch_dct as dct

from torch import nn

from einops import rearrange

from model.modules import EncodingBlock, SA_GC
from model.utils import bn_init, sample_standard_gaussian, import_class
from model.encoder_decoder import Encoder_z0_RNN

from einops import rearrange

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torchdiffeq import odeint as odeint


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol=1e-4,
                 odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
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
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        # self.pos_embedding = nn.Parameter(torch.randn(1, dim, T*2, 1))
        # self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, t, x, backwards=False):
        # TODO:refactroing
        # TODO:temporal embedding
        # t = int(t.item())
        # x = x + self.pos_embedding[:,:,t:t+1]
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.einsum('vu,nctu->nctv', self.A.to(x.device).to(x.dtype), x)
        x = self.conv3(x)
        x = self.relu(x)

        if backwards:
            x = -x
        return x


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, ode_solver_method='rk4',
                 graph=None, in_channels=3, num_head=3, k=0, base_channel=64, device='cuda',
                 dct=True):
        super(InfoGCN, self).__init__()

        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        self.Graph = import_class(graph)()
        A = np.stack([self.Graph.A_norm] * num_head, axis=0)
        self.dct = dct

        self.num_class = num_class
        self.num_point = num_point
        self.A_vector = self.get_A(k)
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        bn_init(self.data_bn, 1)
        ode_func = ODEFunc(base_channel, torch.from_numpy(self.Graph.A_norm), T=64).to(device)

        self.diffeq_solver = DiffeqSolver(ode_func, ode_solver_method,
            odeint_rtol=1e-5, odeint_atol=1e-4, device=device)

        self.encoder = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            SA_GC(base_channel, base_channel, A),
        )
        self.encoder_z0 = Encoder_z0_RNN(base_channel, A, device)

        self.recon_decoder = nn.Sequential(
            SA_GC(base_channel, base_channel, A),
            SA_GC(base_channel, base_channel, A),
            nn.Conv2d(base_channel, 3, 1),
        )

        self.cls_decoder =  nn.Sequential(
            EncodingBlock(base_channel, base_channel, A),
            # EncodingBlock(base_channel, base_channel, A),
            # EncodingBlock(base_channel, base_channel, A),
            # EncodingBlock(base_channel, base_channel*2, A, stride=2),
            # EncodingBlock(base_channel*2, base_channel*2, A),
            # EncodingBlock(base_channel*2, base_channel*2, A),
            # EncodingBlock(base_channel*2, base_channel*4, A, stride=2),
            # EncodingBlock(base_channel*4, base_channel*4, A),
            # EncodingBlock(base_channel*4, base_channel*4, A),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(base_channel, num_class)
        )

        # self.classifier = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(base_channel*4, num_class)
        # )


    def recon_loss(self, x, x_hat):
        recon_loss = (F.mse_loss(x_hat, x, reduce=None) * (x != 0)).sum()\
            / (x != 0).sum()
        return recon_loss


    def KL_div(self, fp_mu, fp_std, kl_coef=1):
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        loss = kldiv_z0.mean()
        # loss = -torch.logsumexp(-kl_coef * kldiv_z0,0)
        return loss

    def get_A(self, k):
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x, t):

        # dct
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)
        x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

        # embedding
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]

        # encoding
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N, v=V)
        x = self.encoder(x)
        fp_mu, fp_std = self.encoder_z0(x, run_backwards=True)
        kl_div = self.KL_div(fp_mu, fp_std)

        # extrapoloation
        # TODO: kl_div
        # fp_enc = sample_standard_gaussian(fp_mu, fp_std)
        z0 = fp_mu
        z = self.diffeq_solver(z0, t)

        # cls_decoding
        # TODO: match the dimmension
        # x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, n=N)
        # x = self.data_bn(x)
        # x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V)
        y = self.cls_decoder(z)
        y = y.view(N, M, y.size(1), -1).mean(3).mean(1)
        y = self.classifier(y)

        # recon_decoding
        x_hat = self.recon_decoder(z)
        x_hat = rearrange(x_hat, '(n m) c t v -> n c t v m', n=N, m=M)
        return y, x_hat, kl_div
