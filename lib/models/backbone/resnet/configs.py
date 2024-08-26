#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:36:09 2021

@author: vivi
"""

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224), 
        'block': 'BasicBlock', 
        'layers': [2, 2, 2, 2],
        'progress': True, 
        **kwargs
    }

default_cfgs = {
    'resnet18': _cfg(
        layers=[2, 2, 2, 2],
        url='https://download.pytorch.org/models/resnet18-f37072fd.pth',
        ),
    'resnet34': _cfg(
        layers=[3, 4, 6, 3],
        url='https://download.pytorch.org/models/resnet34-b627a593.pth',
        ),
    'resnet50': _cfg(
        block='Bottleneck',
        layers= [3, 4, 6, 3],
        url='https://download.pytorch.org/models/resnet50-0676ba61.pth',
        ),
    'resnet101': _cfg(
        block='Bottleneck',
        layers= [3, 4, 23, 3],
        url='https://download.pytorch.org/models/resnet101-63fe2227.pth',
        ),
    'resnet152': _cfg(
        block='Bottleneck',
        layers= [3, 8, 36, 3],
        url='https://download.pytorch.org/models/resnet152-394f9c45.pth',
        ),
    'resnext50_32x4d': _cfg(
        block='Bottleneck',
        layers= [3, 4, 6, 3],
        groups=32,
        width_per_group=4,
        url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        ),
    'resnext101_32x8d': _cfg(
        block='Bottleneck',
        layers= [3, 4, 23, 3],
        groups=32,
        width_per_group=8,
        url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        ),
    r"""Wide ResNet-50-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    'wide_resnet50_2': _cfg(
        block='Bottleneck',
        layers= [3, 4, 6, 3],
        width_per_group=128,
        url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        ),
    'wide_resnet101_2': _cfg(
        block='Bottleneck',
        layers= [3, 4, 23, 3],
        width_per_group=128,
        url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        ),
}