#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:53:49 2021

@author: cimon
"""

import os
import sys
import json
import random
import numpy as np
#from ray import tune
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
# os.path.abspath(__file__)获取当前文本的绝对路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)
from lib.datasets import DataLoaderX 
from lib.datasets.data_loader import SurAnal1D_Dataset, SurAnal2D_Dataset
from lib.models.surv_models import Model_DeepHit, Model_DeepCox, Model_DeepCox_Hazard, Model_MTLR
from lib.models.surv_MRI import SeResNeXt
from lib.models.backbone.resnet.resnet import ResNet
from lib.functions.function import train, validate, testing
from optimizer import get_optimizer
from warmup_scheduler import WarmupScheduler
from config_parser import config_parser
from utils import get_logger, save_checkpoint
from torchvision import datasets, transforms
#from torchsummary import summary

#from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params/1000000

class SurvivalAnalysis():
    def __init__(self, args, **kwargs):
        self.initial_seeds(args.seed)
        self.configs = self.load_configs(args.config)
        self.name = self.configs['name']
        self.model_name = self.configs["model_name"]
        log_dir = os.path.join(self.configs['train']['log_dir'], self.configs["dataset_name"])
        self.log_dir = os.path.join(log_dir,self.configs["model_name"])
        save_dir = os.path.join(self.configs['train']['save_dir'], self.configs["dataset_name"])
        self.save_dir = os.path.join(save_dir, self.configs["model_name"])
        self.logger, self.save_folder, self.writer_dict = self.mkdirs()
        # self.tune = self.configs['tune']['switch'].upper() == 'ON'
    
    def initial_seeds(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    def load_configs(self, config_file):
        config_file = os.path.join(parent_dir, config_file)
        assert os.path.exists(config_file)
        with open(config_file, 'rb') as f:
            configs = json.load(f)
        return configs
    
    def mkdirs(self):
        log_folder = os.path.join(self.log_dir, self.name, 'train_val')
        os.makedirs(log_folder, exist_ok=True)
        logger, tb_folder, time_str = get_logger(log_folder)
        
        save_folder = os.path.join(self.save_dir, self.name, time_str)
        os.makedirs(save_folder, exist_ok=True)
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_folder),
            'train_global_steps': 0,
            'valid_global_steps': 0,
            'test_global_steps': 0
        }
        return logger, save_folder, writer_dict
    
    def get_dataset(self, dataset_name):
        data_arg = self.configs["data"][dataset_name]
        self.configs["data"] = data_arg
        if dataset_name == "mnist":
            data_arg["img_set"] = "train"
            train_dataset = SurAnal2D_Dataset(data_arg)
            self.D_in = train_dataset.D_in
            
            data_arg["img_set"] = "valid"
            val_dataset = SurAnal2D_Dataset(data_arg)
            
            data_arg["img_set"] = "test"
            test_dataset = SurAnal2D_Dataset(data_arg)
            
        elif dataset_name in ["metabric", "bidding", "support", "npc", "npc-mri"]:
            data_arg["img_set"] = "train"
            train_dataset = SurAnal1D_Dataset(data_arg)
            self.D_in = train_dataset.D_in
            
            data_arg["img_set"] = "valid"
            val_dataset = SurAnal1D_Dataset(data_arg)
            
            data_arg["img_set"] = "test"
            test_dataset = SurAnal1D_Dataset(data_arg)

        else:   
            raise Exception("No implementation of dataset generator for {}".format(dataset_name))
            
        return train_dataset, val_dataset, test_dataset
            
    
    def get_dataloader(self):
        dataset_name = self.configs["dataset_name"]
        train_dataset, val_dataset, test_dataset = self.get_dataset(dataset_name)
        self.logger.info("current dataset: {}".format(dataset_name))
        self.train_num = len(train_dataset)
        
        num_workers = self.configs['train']['num_workers']
        batch_size=self.configs['train']['batch_size']
                
        train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=True)

        val_loader = DataLoaderX(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False)

        test_loader = DataLoaderX(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False)

        self.train_num = len(train_dataset)
        self.logger.info("#training cases: {}".format(len(train_dataset)))

        self.valid_num = len(val_dataset)
        self.logger.info("#validation cases: {}".format(len(val_dataset)))

        self.test_num = len(test_dataset)
        self.logger.info("#test cases: {}".format(len(test_dataset)))
        
        # self.configs["loss"]['num_classes'] = self.configs['data']['num_classes']
        # self.logger.info("#diseases: {}".format(self.configs['data']['num_classes']))
        
        return train_loader, val_loader, test_loader
    
    def init_model(self):
        # Model
        model_name = self.configs['model_name']
        self.logger.info("current model: {}".format(model_name))
        self.configs['network'] = self.configs['network'][model_name]
        self.configs['network']['D_in'] = self.D_in

        net_args = self.configs['network']
        if model_name == "deep_hit":
            model = Model_DeepHit(**net_args)
        elif model_name == "deep_cox":
            model = Model_DeepCox(**net_args)
        elif model_name == "deep_cox_hazard":
            model = Model_DeepCox_Hazard(**net_args)
        elif model_name == "deep_mtlr":
            model = Model_MTLR(**net_args)
        elif model_name == "resnet":
            model = ResNet(**net_args)
        elif model_name == "senet":
            model = SeResNeXt(**net_args)
        else:
            raise Exception("No implementation of deep learning model for {}".format(model_name))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        num_params = count_parameters(model)
        self.logger.info("#Total Parameter: \t%4.3fM" % num_params)
        # summary(model, (self.configs["data"]["in_chans"], self.configs["data"]["input_size"], self.configs["data"]["input_size"]))
        # import pdb;pdb.set_trace()
        learned_dict = []
        # print("model parameters:")
        for name, param in model.named_parameters():
            # print("{}, size: {}, is grad: {}".format(name, param.size(), param.requires_grad))
            if param.requires_grad:
                learned_dict.append(name)
        
        finetune = False
        if self.configs['pretrained']:
            pretrained = self.configs['pretrained'] # using absolute path
            self.logger.info('pretrained:{}'.format(pretrained))
            state_dict = torch.load(pretrained, map_location='cpu')
            if "model" in state_dict.keys():
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)

        if finetune:
            # 为了更快获得模型，只进行微调即可
            print("to do someting")
            # para_list = ["head.weight", "head.bias", "cls_token", "layers.1.downsample.convert.weight", 'layers.1.downsample.norm_cls.weight', 'layers.1.downsample.norm_cls.bias', "layers.2.downsample.convert.weight", 'layers.2.downsample.norm_cls.weight', 'layers.2.downsample.norm_cls.bias', "layers.3.downsample.convert.weight", 'layers.3.downsample.norm_cls.weight', 'layers.3.downsample.norm_cls.bias']
            # self.logger.info("finetune: only update weight {}".format(para_list))
            # for name, param in model.named_parameters():
            #     if name not in para_list:
            #         param.requires_grad = False
                
        num_gpus = self.configs['train']['num_gpus']
        self.logger.info("Using {} GPUs".format(num_gpus))
        model = nn.DataParallel(model, device_ids=list(range(num_gpus))).cuda()
        model.train()
                
        return model
    
    def init_optimizer(self):
        optimizer = get_optimizer(self.model.parameters(), self.configs['train']['optimizer'])
        lr_scheduler = WarmupScheduler(optimizer, 
                                       total_epochs=self.configs['train']['epochs'],
                                       warmup_param=self.configs['train']['scheduler']['warmup'],
                                       scheduler_param=self.configs['train']['scheduler'])
        return optimizer, lr_scheduler
    
    
    def run(self):
        # if self.tune:
        #      #tune.utils.wait_for_gpu()
        #      self.update_tune_params()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()
        
        self.model = self.init_model()
        self.optimizer, self.lr_scheduler = self.init_optimizer()
        self.logger.info("configs:\n {}".format(self.configs))
        dataset_name = self.configs['dataset_name']
        self.logger.info("begin training using loss: {} and model: {}".format(self.configs['loss_list'],self.configs['model_name']))
        pre_perf, cur_perf, best_perf, best_avg_perf = 0.45, 0.45, 0.45, 0.45
        max_epochs = self.configs['train']['epochs']
        for epoch in range(0, max_epochs):
            print("begin training")
            # import pdb;pdb.set_trace()
            # train for one epoch
            pre_perf = cur_perf
            best_model, best_avg = False, False
            grad_accum_step = self.configs['train']['grad_accum_step']
            train_loss,cindex = train(self.configs['model_name'], self.configs['loss_list'], self.train_loader, self.model, epoch, self.optimizer, los_args = self.configs["loss"],
                               logger=self.logger, grad_accum_step=grad_accum_step, writer_dict=self.writer_dict)
            # evaluate on validation set
            tra_c = validate(self.configs['model_name'], self.configs['loss_list'], self.train_loader, self.model,self.logger, los_args = self.configs["loss"], writer_dict=self.writer_dict)
            cur_perf = validate(self.configs['model_name'], self.configs['loss_list'], self.val_loader, self.model,self.logger, los_args = self.configs["loss"], writer_dict=self.writer_dict)

            testing(self.configs['model_name'], self.configs['loss_list'], self.test_loader, self.model, self.logger, los_args = self.configs["loss"], writer_dict=self.writer_dict)
            # import pdb;pdb.set_trace()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch=epoch, metrics=train_loss)
            
            # sigma = 10.0/(epoch+1.0)
            # sigma = max(1.0,sigma)
            # cur_perf = (cur_perf+sigma*cindex)/(1+sigma)
            avg_perf = (pre_perf + cur_perf) / 2
            if epoch >= max_epochs/2 and epoch%10==0:
                if tra_c>0.53 and cur_perf >= best_perf:
                    best_perf = cur_perf
                    best_model = True
                elif avg_perf >= best_avg_perf:
                    best_avg_perf = avg_perf
                    best_avg = True
                else:
                    pass
            if best_model:  # or best_avg or (epoch+1)==max_epochs: # or (epoch+1)%100==0
                self.logger.info("=> Best: {:.4f}".format(best_perf))  # if best_model else best_avg_perf))
                self.logger.info('=> saving checkpoint to {}'.format(self.save_folder))
                state_dict = self.model.module.state_dict()
                # embed_mean = self.model.module.embed_mean if self.configs['network']['embed'] else None
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': self.model_name,
                    'state_dict': state_dict,
                    'perf': best_perf if best_model else best_avg_perf,
                    'optimizer': self.optimizer.state_dict(),
                }, best_model, self.save_folder)


        self.writer_dict['writer'].close()
        self.logger.info('=> the best result in the validation: {}'.format(best_perf))



if __name__ == '__main__':
    args = config_parser()
    clss = SurvivalAnalysis(args)
    clss.run()
    
