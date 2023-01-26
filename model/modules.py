import math

import torch
import numpy as np

from torch import nn, einsum
from einops import rearrange
from torch.autograd import Variable
from einops.layers.torch import Rearrange

from model.utils import conv_init, bn_init, conv_branch_init


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()
        return attn


class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)


    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        out = None
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :] # (nt)vv
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GCN, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

    def forward(self, x):
        N, C, T, V = x.size()

        out = None
        A = self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :] # (nt)vv
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, seq_len, heads=8, dim_head=64, dropout=0.,
                 use_mask=False, SAGC_proj=False, A=1, num_point=25):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_mask = use_mask
        self.num_point = num_point

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = SA_GC(dim, inner_dim * 3, A) if SAGC_proj else\
            GCN(dim, inner_dim * 3, A)

        self.register_buffer("mask", torch.ones(seq_len, seq_len).tril()
                                     .view(1, 1, seq_len, seq_len))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape; V= self.num_point

        x = rearrange(x, '(b v) t c -> b c t v', v=V)
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b c t v -> (b v) t c', v=V)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # (N*V*M, H, T, D)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.use_mask:
            attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, max_seq_len,
                 dropout=0., use_mask=True, A=1, num_point=25, SAGC_proj=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(
                    dim, max_seq_len, heads=heads, dim_head=dim_head,
                    dropout=dropout, use_mask=use_mask, SAGC_proj=SAGC_proj, A=A, num_point=num_point)
                ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self._attns = []

    def forward(self, x):
        attns = []
        for attn, ff in self.layers:
            res = x
            x, sa = attn(x)
            x += res
            x = ff(x) + x
            attns.append(sa.clone())
        self._attns = attns
        return x

    def get_attns(self):
        return self._attns


def PositionalEncoding(d_model: int, dropout: float = 0.1, max_len: int = 5000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


class TemporalEncoder(nn.Module):
    def __init__(self, seq_len, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0., A=1, num_point=25, SAGC_proj=True, device='cuda'):
        super().__init__()

        self.pe = PositionalEncoding(d_model=dim, max_len=seq_len).to(device)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, seq_len, dropout, A=A, num_point=num_point, use_mask=True, SAGC_proj=SAGC_proj)
        self.to_latent = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        in : b t c h w
        out : b t c h w
        """
        B, C, T, V = x.shape
        x = rearrange(x, 'b c t v -> (b v) t c')
        x = x + self.pe[:, :T, :]
        x = self.transformer(x)
        x = self.to_latent(x)
        x = rearrange(x, '(b v) t c -> b c t v', v=V)
        return x

    def get_attention(self):
        return self.transformer._attns
