#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:21:46 2021

@author: vivi
"""

import torch.nn as nn

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'mean': (0.5, 0.5, 0.5), 
        'std': (0.5, 0.5, 0.5),
        'input_size': (3, 224, 224), 
        'patch_size': 16, 
        'embed_dim': 768, 
        'depth': 12, 
        'num_heads': 12,
        'pool_size': None,
        'crop_pct': 0.9, 
        'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models (my experiments)

    'vit_tiny_patch16_224?': _cfg(
        depth=12,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=3.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),


    'vit_small_patch16_224': _cfg(
        depth=8, 
        num_heads=8, 
        mlp_ratio=3.0,
        qkv_bias=False, 
        norm_layer=nn.LayerNorm,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    
    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        patch_size=32, 
        ),
    'vit_base_patch16_384': _cfg(
        input_size=(3, 384, 384), 
        crop_pct=1.0,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth'
        ),
    'vit_base_patch32_384': _cfg(
        patch_size=32, 
        input_size=(3, 384, 384), 
        crop_pct=1.0,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        ),
    'vit_large_patch16_224': _cfg(
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        ),
    'vit_large_patch32_224': _cfg(
        patch_size=32, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch16_384': _cfg(
        embed_dim=1024,
        input_size=(3, 384, 384),
        depth=24, 
        num_heads=16,
        crop_pct=1.0,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        ),
    'vit_large_patch32_384': _cfg(
        input_size=(3, 384, 384),
        patch_size=32, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        crop_pct=1.0,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        ),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        representation_size=768,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        ),
    'vit_base_patch32_224_in21k': _cfg(
        patch_size=32,
        representation_size=768,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        ),
    'vit_large_patch16_224_in21k': _cfg(
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        representation_size=1024,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        ),
    'vit_large_patch32_224_in21k': _cfg(
        patch_size=32, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        representation_size=1024,
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        ),
    'vit_huge_patch14_224_in21k': _cfg(
        patch_size=14, 
        embed_dim=1280, 
        depth=32, num_heads=16, 
        representation_size=1280,
        url='',  # FIXME I have weights for this but > 2GB limit for github release binaries
        ),
    '''
    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        hybrid_backbone=backbone_resnet,
        representation_size=768,
        first_conv='patch_embed.backbone.stem.conv',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        ),
    'vit_base_resnet50_384': _cfg(
        hybrid_backbone=backbone_resnet,
        input_size=(3, 384, 384),
        crop_pct=1.0, 
        first_conv='patch_embed.backbone.stem.conv',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        ),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(
        depth=8, 
        num_heads=8, 
        mlp_ratio=3, 
        hybrid_backbone=backbone_resnet26d
        ),
    'vit_small_resnet50d_s3_224': _cfg(
        depth=8, 
        num_heads=8, 
        mlp_ratio=3, 
        hybrid_backbone=backbone_resnet50d_3
        ),
    'vit_base_resnet26d_224': _cfg(
        depth=8, 
        num_heads=8, 
        mlp_ratio=3, 
        hybrid_backbone=backbone_resnet26d
        ),
    'vit_base_resnet50d_224': _cfg(
        depth=8, 
        num_heads=8, 
        mlp_ratio=3, 
        hybrid_backbone=backbone_resnet50d_4
        ),
    '''
    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        embed_dim=192, 
        num_heads=3,
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        embed_dim=384,
        num_heads=6,
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        input_size=(3, 384, 384),
        crop_pct=1.0,
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        ),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        input_size=(3, 384, 384), 
        crop_pct=1.0,
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        ),
}