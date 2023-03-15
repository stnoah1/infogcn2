import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import linalg as LA

def masked_recon_loss(x, x_hat, mask):
    recon_loss = (F.mse_loss(x_hat, x, reduction="none") * mask).sum()\
        / mask.sum()
    return recon_loss

def cosine_similarity(x, x_hat, mask):
    recon_loss = (1 - F.cosine_similarity(x - x[:,:,:,1:2], x_hat - x_hat[:,:,:,1:2], dim=1) * mask[:,0]).sum()\
        / mask.sum()
    return recon_loss

class ReconLoss(nn.Module):
    def __init__(self, p=2):
        super(ReconLoss, self).__init__()
        self.loss = nn.PairwiseDistance(p)

    def forward(self, pred, gt):
        B, V, C = pred.shape
        loss = self.loss(pred.contiguous().view(-1, C), gt.contiguous().view(-1, C))
        return loss.view(B, V).mean(-1).mean(-1)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, T=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

