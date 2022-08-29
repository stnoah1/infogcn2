import numpy as np
import torch
import torch.nn.functional as F

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
        pred_y = rearrange(pred_y, 't b c e v -> b t c v e').squeeze()

        return pred_y


class ODEFunc(nn.Module):
    def __init__(self, dim, num_layer):
        super(ODEFunc, self).__init__()
        layers = []
        for i in range(num_layer):
            layers.append(nn.Conv2d(dim, dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, t, x, bacwards=False):
        grad = self.layers(x)
        if bacwards:
            grad = -grad
        return grad


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, obs_p=0.1, ode_solver_method=None,
                 graph=None, in_channels=3, num_head=3, k=0, device='cuda'):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)
        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.A_vector = self.get_A(graph, k)
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        # self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        # bn_init(self.data_bn, 1)
        ode_func = ODEFunc(base_channel, 4)

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
            SA_GC(base_channel, 3, A),
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
            nn.Linear(base_channel*4, num_class)
        )

    def recon_loss(self, x, x_hat):
        # todo: mask empty
        return F.mse_loss(x, x_hat, reduction='mean')


    def KL_div(self, fp_mu, fp_std, kl_coef=1):
        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        loss = kldiv_z0.mean((3,4)).sum(0).mean(0).mean(0) # hw -> mean , t -> sum, b,c -> mean
        # loss = -torch.logsumexp(-kl_coef * kldiv_z0,0)
        return loss

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x, t):

        # data preparation
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V)
        x = self.A_vector.to(x.device).to(x.dtype).expand(N*M*T, -1, -1) @ x

        # embedding
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]

        # encoding
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, t=T)
        x = self.encoder(x)
        fp_mu, fp_std = self.encoder_z0(x)

        # extrapoloation
        # TODO: kl_div
        # fp_enc = sample_standard_gaussian(fp_mu, fp_std)
        fp_enc = fp_mu
        z = self.diffeq_solver(fp_enc, t)

        # cls_decoding
        y = self.cls_decoder(z)
        y = y.view(N, M, y.size(1), -1).mean(3).mean(1)
        y = self.classifier(y)

        # recon_decoding
        x_hat = self.recon_decoder(z)
        x_hat = rearrange(x_hat, '(n m) c t v -> n c t v m', m=M)

        return y, x_hat
