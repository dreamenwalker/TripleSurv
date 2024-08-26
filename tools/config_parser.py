#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:43:09 2020

@author: cimon
"""

import argparse

def config_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Skin Diseases Classifier')
    # common
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed')
    parser.add_argument('--config', type=str, default='configs/train.json', #swin__7pt.json
                        help='config file path')
    parser.add_argument('--root', type=str, default='data', 
                        help='image folder')
    parser.add_argument('--save-dir', type=str, default='output', 
                        help='image folder')
    parser.add_argument('--log-dir', type=str, default='log', 
                        help='image folder')
    parser.add_argument('--pretrained', type=str, default="", 
                        help='pretrained parameters')
    parser.add_argument('--mode', type=str, default="train", 
                        help='train or eval')
    
    args = parser.parse_args()
    return args


def update_configs(configs,  args):
    #TODO
    pass
    