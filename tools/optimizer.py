#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:52:43 2021

@author: vivi
https://www.jianshu.com/p/39dac1e24709
"""

import torch.optim as optim

def get_optimizer(parameters, optm_params):
    if optm_params['type'] == 'SGD':
        optimizer = optim.SGD(parameters, 
                              lr=optm_params['lr'], 
                              momentum=optm_params['momentum'], 
                              weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'Adam':
        optimizer = optim.Adam(parameters, 
                               lr=optm_params['lr'], 
                               weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'AdamW':
        optimizer = optim.AdamW(parameters, 
                                lr=optm_params['lr'], 
                                weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'Adamax':
        optimizer = optim.Adamax(parameters, 
                                 lr=optm_params['lr'], 
                                 weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'SparseAdam':
        optimizer = optim.SparseAdam(parameters, 
                                     lr=optm_params['lr'])
    elif optm_params['type'] == 'Adagrad':
        optimizer = optim.Adagrad(parameters, 
                                  lr=optm_params['lr'],
                                  lr_decay=optm_params['lr_decay'],
                                  weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'Adadelta':
        optimizer = optim.Adadelta(parameters, 
                                   lr=optm_params['lr'], 
                                   weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'RMSprop':
        optimizer = optim.RMSprop(parameters, 
                                  lr=optm_params['lr'], 
                                  momentum=optm_params['momentum'], 
                                  weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'Rprop':
        optimizer = optim.Rprop(parameters, 
                                lr=optm_params['lr'], 
                                etas=optm_params['etas'], 
                                step_sizes=optm_params['step_sizes'])
    elif optm_params['type'] == 'ASGD':
        optimizer = optim.ASGD(parameters, 
                               lr=optm_params['lr'], 
                               weight_decay=optm_params['weight_decay'])
    elif optm_params['type'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, 
                                lr=optm_params['lr'], 
                                max_iter=optm_params['max_iter'],
                                history_size=optm_params['history_size'])
    else:
        raise ValueError("unsupported optimizer: {}".format(optm_params['type']))
    
    return optimizer