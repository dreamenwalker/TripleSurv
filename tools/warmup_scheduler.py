#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:43:09 2020

@author: cimon
"""

import math
from torch.optim import lr_scheduler  

class WarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs=-1, 
                 warmup_param=None, scheduler_param=None,
                 last_epoch=-1, verbose=False):
        self.warmup_param = warmup_param
        self.scheduler_param = scheduler_param
        self.total_epochs = total_epochs
        self.lrs = self.scheduler(optimizer)
        super(WarmupScheduler, self).__init__(optimizer)
    
    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)
        
    
    def warmup(self, step):
        if self.warmup_param['type'] == 'constant':
            return self.warmup_param['start']
        elif self.warmup_param['type'] == 'linear':
            return ((1.0 - self.warmup_param['start']) / self.warmup_param['steps']) \
                * step + self.warmup_param['start']
        elif self.warmup_param['type'] == 'exponential':
            return math.exp((math.log(2-self.warmup_param['start'])/self.warmup_param['steps']) * step) \
                + self.warmup_param['start'] - 1.0
        else:
            raise ValueError("unsupported warmup type: {}".format(self.warmup_param['type']))
    
    def scheduler(self, optimizer):
        if self.scheduler_param['type'] == 'step':
            lrs = lr_scheduler.StepLR(optimizer, 
                                      step_size=self.scheduler_param['step'],
                                      gamma=self.scheduler_param['gamma'])
        elif self.scheduler_param['type'] == 'multistep':
            lrs = lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=self.scheduler_param['milestones'],
                                           gamma=self.scheduler_param['gamma'])
        elif self.scheduler_param['type'] == 'exponential':
            lrs = lr_scheduler.ExponentialLR(optimizer, 
                                             gamma=self.scheduler_param['gamma'])
        elif self.scheduler_param['type'] == 'cosine':
            lrs = lr_scheduler.CosineAnnealingLR(optimizer, 
                                                 T_max=self.scheduler_param['step'],
                                                 eta_min=self.scheduler_param['min'])
        elif self.scheduler_param['type'] == 'cosinewarm':
            lrs = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                           T_0=self.scheduler_param['t0'],
                                                           T_mult=self.scheduler_param['factor'],
                                                           eta_min=self.scheduler_param['min'])
        elif self.scheduler_param['type'] == 'cycle':
            lrs = lr_scheduler.CyclicLR(optimizer, 
                                        base_lr=self.scheduler_param['min'],
                                        max_lr=self.scheduler_param['max'],
                                        step_size_up=self.scheduler_param['step'])
        elif self.scheduler_param['type'] == 'onecycle':
            lrs = lr_scheduler.OneCycleLR(optimizer, 
                                          max_lr=self.scheduler_param['max'],
                                          total_steps=self.total_epochs,
                                          pct_start=self.scheduler_param['percent'],
                                          anneal_strategy='cos',
                                          div_factor=self.scheduler_param['max']//self.scheduler_param['start'],
                                          final_div_factor=self.scheduler_param['max']//self.scheduler_param['min'])
        elif self.scheduler_param['type'] == 'plateau':
            lrs = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode=self.scheduler_param['mode'],
                                                 factor=self.scheduler_param['factor'],
                                                 patience=self.scheduler_param['patience'],
                                                 threshold=self.scheduler_param['threshold'],
                                                 min_lr=self.scheduler_param['min'])
        else:
            raise ValueError("unsupported scheduler type: {}".format(self.scheduler_param['type']))
        
        return lrs
    
    
    def get_lr(self):
        # print("last_epoch: {} -> lr: {}".format(self.last_epoch, lr))
        if self.last_epoch < self.warmup_param['steps']:
            multiplier = self.warmup(self.last_epoch)
            print('warmup',multiplier,self.base_lrs)
            return [base_lr * multiplier for base_lr in self.base_lrs]
        else:
             return self.lrs.get_last_lr()


    def step(self, epoch=None, metrics=None):
        if epoch is None:
            if self.last_epoch == -1:
                self.last_epoch = epoch = 0
            else:
                epoch = self.last_epoch+1
        self.last_epoch = epoch 
        print("self.last_epoch:", self.last_epoch)
        if self.last_epoch < self.warmup_param['steps']:
            return super(WarmupScheduler, self).step(epoch)
        else:
            if self.scheduler_param['type'] != 'plateau':
                self.lrs.step(epoch - self.warmup_param['steps'])
            else:
                self.lrs.step(metrics, epoch-self.warmup_param['steps'])


if __name__ == '__main__':
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    t = torch.tensor([0.0], requires_grad=True)
    optimier = torch.optim.SGD([t], lr=0.1)
    epochs=50
    warmup_prams={"steps": 5,
                  "type":"linear",
                  "start":0.001}
    sheduler_param={"type": "cosine",
                    "step": 45,
                    "percent": 0.1,
                    "max": 0.1,
                    "start": 0.001,
                    "min": 0.00001}
    wm_scheduler = WarmupScheduler(optimier,epochs,warmup_prams,sheduler_param)
    res = []
    for epoch in range(epochs):
        optimier.zero_grad()
        optimier.step()
        wm_scheduler.step(epoch)
        res.append([epoch, optimier.param_groups[0]['lr']])
        print(epoch, ":", optimier.param_groups[0]['lr'])
    res = np.asarray(res)
    plt.plot(res[:,0], res[:,1])
    plt.show()