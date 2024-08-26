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
from ray import tune
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
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
from lib.models.backbone.resnet.resnet import ResNet
from optimizer import get_optimizer
from warmup_scheduler import WarmupScheduler
from config_parser import config_parser
from utils import get_logger, save_checkpoint
from torchvision import datasets, transforms
from torchsummary import summary
import pandas as pd
#from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import collections
from lib.utils.util import get_risk, get_surv_mask1, get_surv_mask2
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings("ignore")



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params/1000000

def get_loss(preds, Y_label_T, Y_label_E, key, mydict):
    if key == "likelihood":
        surv_mask1 = get_surv_mask1(Y_label_T, Y_label_E, num_Category = mydict['Kbins'])
        return mydict["gamma"] * likelihood_loss(preds, surv_mask1)

    elif key == "partial_likelihood":
        if mydict["mode"] == "breslow":
            return mydict["gamma"] * partial_likelihood_loss_breslow(preds, Y_label_T, Y_label_E)
        else:
            return mydict["gamma"] * partial_likelihood_loss_efron(preds, Y_label_T, Y_label_E)

    elif key == "rank":
        if mydict['mode'] == "ours":
            return mydict["gamma"] * ranking_loss(preds, Y_label_T, Y_label_E, sigma=mydict["sigma"])
        elif mydict['mode'] == "hit":
            surv_mask2 = get_surv_mask2(Y_label_T, num_Category = mydict['Kbins'])
            return mydict["gamma"] * ranking_loss_hit(preds, Y_label_T, Y_label_E, surv_mask2, sigma=mydict["sigma"])
        else:
            raise Exception("No implementation of the rank loss: {}".format(mydict['mode']))

    elif key == "self_rank":
        return mydict["gamma"] * self_ranking_loss_weight(preds, Y_label_T, Y_label_E, sigma=mydict["sigma"], scale=mydict["scale"])

    elif key == "calibration":
        surv_mask1 = get_surv_mask1(Y_label_T, Y_label_E, num_Category = mydict['Kbins'])
        surv_mask2 = get_surv_mask2(Y_label_T, num_Category = mydict['Kbins'])
        return mydict["gamma"] *  calibration_loss(preds, Y_label_E, surv_mask1, surv_mask2, nbins=mydict["nbins"])

    else:
         raise Exception("No implementation of the loss: {}".format(key))
def load_configs(config_file):
    config_file = os.path.join(parent_dir, config_file)
    assert os.path.exists(config_file)
    with open(config_file, 'rb') as f:
        configs = json.load(f)
    return configs
class SurvivalAnalysis():
    def __init__(self, args, config, **kwargs):
        self.initial_seeds(args.seed)
        self.configs = config
        self.name = self.configs['name']
        self.model_name = self.configs["model_name"]
        output_path = os.path.join(self.configs['output_path'], self.configs["dataset_name"])
        output_path = os.path.join(output_path, self.configs["model_name"])
        self.output_path = os.path.join(output_path,self.name)
        os.makedirs(self.output_path, exist_ok=True)
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
        # os.makedirs(save_folder, exist_ok=True)
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
            
        elif dataset_name in ["metabric", "bidding", "support", "npc"]:
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
        print("current dataset: {}".format(dataset_name))
        self.train_num = len(train_dataset)
        
        num_workers = self.configs['train']['num_workers']
        batch_size=self.configs['train']['batch_size']
                
        train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False)

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
        print("#training cases: {}".format(len(train_dataset)))

        self.valid_num = len(val_dataset)
        print("#validation cases: {}".format(len(val_dataset)))

        self.test_num = len(test_dataset)
        print("#test cases: {}".format(len(test_dataset)))
        
        # self.configs["loss"]['num_classes'] = self.configs['data']['num_classes']
        # print("#diseases: {}".format(self.configs['data']['num_classes']))
        
        return train_loader, val_loader, test_loader
    
    def init_model(self):
        # Model
        model_name = self.configs['model_name']
        print("current model: {}".format(model_name))
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
        else:
            raise Exception("No implementation of deep learning model for {}".format(model_name))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        num_params = count_parameters(model)
        print("#Total Parameter: \t%4.3fM" % num_params)
        # summary(model, (self.configs["data"]["in_chans"], self.configs["data"]["input_size"], self.configs["data"]["input_size"]))
        
        if self.configs['pretrained']:
            pretrained = self.configs['pretrained'] # using absolute path
            print('pretrained:{}'.format(pretrained))
            state_dict = torch.load(pretrained, map_location='cpu')
            if "model" in state_dict.keys():
                state_dict = state_dict['model']
            # import pdb;pdb.set_trace()
            model.load_state_dict(state_dict, strict=False)
                
        num_gpus = 1
        model = nn.DataParallel(model, device_ids=list(range(num_gpus))).cuda()
        model.eval()
                
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
        print("configs:\n {}".format(self.configs))
        dataset_name = self.configs['dataset_name']
        self.validate(self.configs['model_name'], self.train_loader, self.model, self.configs,mode="train")
        self.validate(self.configs['model_name'], self.val_loader, self.model, self.configs,mode="valid")
        self.validate(self.configs['model_name'], self.test_loader, self.model, self.configs)

    def validate(self, model_name, val_loader, model, config, mode=""):
        # switch to evaluate mode
        model.eval()
        print("-"*100)
        surv_time = []
        surv_status = []
        pred_risk = []
        pred_values = []
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                images,Y_label_T, Y_label_E = data[0], data[1], data[2]
                Y_label_T = Y_label_T.unsqueeze(1)
                Y_label_E = Y_label_E.unsqueeze(1)
                num_images = images.size(0)
                surv_time.append(Y_label_T)
                surv_status.append(Y_label_E)
                # compute output
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    Y_label_T = Y_label_T.cuda(non_blocking=True)
                    Y_label_E = Y_label_E.cuda(non_blocking=True)

                preds = model(images)
                pred_values.append(preds)
                if model_name == "deep_cox":
                    risks = preds
                else:
                    risks = get_risk(preds)
                pred_risk.append(risks)
        surv_time = torch.cat(surv_time, dim=0)
        surv_status = torch.cat(surv_status, dim=0)
        pred_risk = torch.cat(pred_risk, dim=0)
        pred_values = torch.cat(pred_values, dim=0)
        surv_time = surv_time.detach().cpu().numpy()
        pred_risk = pred_risk.detach().cpu().numpy()
        surv_status = surv_status.detach().cpu().numpy()
        pred_values = pred_values.detach().cpu().numpy()
        
        c_index = concordance_index(surv_time, 1.0-pred_risk, surv_status)
        print("C-index: {}".format(c_index))
        data = pd.DataFrame(pred_values)
        data["pred_risk"] = pred_risk
        fold = config["data"]["fold"]
        data["fold"] = fold
        data["time"] = surv_time
        data["dead"] = surv_status
        output_path = os.path.join(self.output_path, f"{mode}{fold}_model_outputs.csv")
        # import pdb; pdb.set_trace()
        data.to_csv(output_path, index=False)
        
    


if __name__ == '__main__':
    args = config_parser()
    config = load_configs(args.config)
    # clss = SurvivalAnalysis(args, config)
    # clss.run()
    model_name = "deep_mtlr"
    dataset_name = "bidding"
    name = ["sgd+likelihood","sgd+HitRank2.0","sgd+SelfRank1.0","sgd+0.5likelihood_1.0HitRank2.0","sgd+0.5likelihood_1.0SelfRank1.0","sgd+0.5likelihood_1.0SelfRank1.0_5.0cal"]
    pretrained = ["./triple_loss/model/bidding/deep_mtlr/0_sgd+likelihood/2023-01-15-14-16/model_best.pth",
]
    # name = ["sgd+0.5likelihood_1.0SelfRank1.0"]
    # pretrained = ["./triple_loss/model/bidding/deep_hit/0_sgd+0.5likelihood_1.0SelfRank1.0/2023-01-15-11-39/model_best.pth"]
    config["model_name"] = model_name
    config["dataset_name"] = dataset_name
    assert len(pretrained)==len(name), "Have an error"
    # config["name"] = name
    for i in range(len(pretrained)):
        config["pretrained"] = pretrained[i]
        # config["data"][dataset_name]["fold"] = i
        config["name"] = name[i]
        clss = SurvivalAnalysis(args, config.copy())
        clss.run()
        print(50*"*")
    
