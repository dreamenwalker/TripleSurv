#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# adopt from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn

def Get_act(name):
    if name == "relu":
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "tanh":
        act = nn.Tanh()
    elif name == "softmax":
        act = nn.Softmax(dim=-1)
    elif name == "sigmoid":
        act = nn.Sigmoid()
    else:
        raise Exception("No implementation of the activation for {}".format(name))
    return act
    

class DeepHit(nn.Module):
    def __init__(self,D_in, num_cat_bins, out_layer='softmax'):
        super(DeepHit, self).__init__()
        self.head = nn.Linear(D_in, num_cat_bins)
        self.out_act = Get_act(out_layer)
    def forward(self, src):
        pred = self.out_act(self.head(src))
        return pred

class DeepCox(nn.Module):
    def __init__(self,D_in, num_cat_bins=1, out_layer='tanh'):
        super(DeepCox, self).__init__()
        self.head = nn.Linear(D_in, num_cat_bins)
        self.out_act = Get_act(out_layer)
    def forward(self, src):
        pred = self.out_act(self.head(src))
        return pred

class DeepMTLR(nn.Module):
    def __init__(self,D_in, num_cat_bins, out_layer='softmax'):
        super(DeepMTLR, self).__init__()
        self.head = nn.Linear(D_in, num_cat_bins)
        self.out_act = Get_act(out_layer)
        
    def forward(self, src):
        out = self.head(src)
        out_concat = torch.cat([torch.zeros_like(out[:, :1]), out], dim=-1)
        out = torch.sum(out, dim=-1, keepdim=True) - torch.cumsum(out_concat, dim=-1)
        if torch.isnan(out).any():
            import pdb;pdb.set_trace()
        out = self.out_act(out)

        return out
        
class SeResNeXt(nn.Module):
    r"""ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    def __init__(self, D_in = 513, mode="cat", num_cat_bins=50, out_layer="softmax", **kwargs):
        super(ResNet, self).__init__()
        
        if mode == "cat":
            self.head = DeepHit(D_in, num_cat_bins, out_layer=out_layer)
        elif mode == "cox":
            self.head = DeepCox(D_in, out_layer=out_layer)
        elif mode == "mtlr":
            self.head = DeepMTLR(D_in, num_cat_bins-1, out_layer=out_layer)
        else:
            raise Exception("No implementation of the output for {}".format(mode))
            
    def forward(self, x):
        x = self.head(x)
        return x