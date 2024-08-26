import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms

import pdb

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SurAnal1D_Dataset(Dataset):
    def __init__(self, config):
        if config['img_set'] == 'train':
            suffix = 'tr'
        elif config['img_set'] == 'valid':
            suffix = 'va'
        elif config['img_set'] == 'test':
            suffix = 'te'
        else:
            assert False, "wrong phase"
        dataset = config['dataset']
        fold = config['fold']
        filename = config['filename']
        datafilename = os.path.join(config['root_dir'], f'{dataset}/{filename}_{fold}_{suffix}.pt')
        data = torch.load(datafilename)

        self.src = data[:,0:-2].float()
        self.time_list = data[:, -2].float()
        self.dead_list = data[:, -1].float()
        # import pdb;pdb.set_trace()
        self.D_in = self.src.size()[1]
        self.phase = config['img_set']


    def __getitem__(self, index):
        x = self.src[index]
        y_time = self.time_list[index]
        y_status = self.dead_list[index]
        return x, y_time, y_status

    def __len__(self):
        return self.src.size(0)
        
        
class SurAnal2D_Dataset(Dataset):
    def __init__(self, config):
    
        dataset = config['dataset']
        dataset_lw = dataset.lower()
        img_set = config['img_set']
        datafilename = os.path.join(config['root_dir'], f'{dataset}/{dataset_lw}_{img_set}_data.pt')
        data = torch.load(datafilename)
        datafilename = os.path.join(config['root_dir'], f'{dataset}/{dataset_lw}_{img_set}_survival.pt')
        labels = torch.load(datafilename)

        self.src = data.float()
        self.time_list = labels[:, -2].float()
        self.dead_list = labels[:, -1].float()
        # import pdb;pdb.set_trace()
        self.D_in = 512
        self.phase = config['img_set']
        
        if isinstance(config['input_size'], list):
            self.input_size = tuple(config['input_size'])
        else:
            self.input_size = (config['input_size'], config['input_size'])
            
        if self.phase == 'train':
            self.transform = self.get_train_transform()
        else:
            self.transform = self.get_test_transform()
        # import pdb;pdb.set_trace()

    def get_train_transform(self):
        trans = transforms.Compose([
            transforms.Resize(self.input_size),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.ToTensor(),
            transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            # transforms.Normalize(mean=self.mean, std=self.std)
        ])

        return trans
    
    def get_test_transform(self):
        trans = transforms.Compose([
            transforms.Resize(self.input_size),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.ToTensor(),
            transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            # transforms.Normalize(mean=self.mean, std=self.std)
        ])

        return trans
        
    def __getitem__(self, index):
        img = self.src[index]
        img = img.unsqueeze(0)
        img = img.expand(3,-1,-1)
        img = self.transform(img)
        y_time = self.time_list[index]
        y_status = self.dead_list[index]
        return img, y_time, y_status

    def __len__(self):
        return self.src.shape[0]