#!/usr/bin/env python
# encoding: utf-8
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class MagMarginProduct(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """
    def __init__(self, in_features, out_features, scale=32.0, easy_margin=False):
        super(MagMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin


        # Params
        self.l_margin = 0.45
        self.u_margin = 0.80
        self.l_a = 10
        self.u_a = 110

    
    def _margin(self, x):
        margin = (self.u_margin - self.l_margin) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin
    

    def forward(self, embeddings):
        """
        Here m is a function which generate adaptive margin
        """
        norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self._margin(norms)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(embeddings), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)

        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta
        return [cos_theta, cos_theta_m], norms
    


class MagLoss(torch.nn.Module):
    """
    MagFace Loss.
    """
    def __init__(self):
        super(MagLoss, self).__init__()
        self.u_a = 110

    def calc_loss_G(self, x_norm):
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        return loss.mean(), loss_g, output