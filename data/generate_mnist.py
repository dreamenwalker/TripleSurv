import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
# from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Uniform = torch.distributions.Uniform
Gamma = torch.distributions.Gamma
Kbins = 20
travalset = datasets.MNIST('MNIST/training',train=True,transform=transforms.ToTensor(),download=True)
testset = datasets.MNIST('MNIST/testing',train=False,transform=transforms.ToTensor(),download=True)
train_val_data, train_val_label = travalset.data, travalset.targets
train_data, val_data = train_test_split(train_val_data,test_size = 0.2,random_state = 0)
train_label, val_label = train_test_split(train_val_label,test_size = 0.2,random_state = 0)
test_data, test_label = testset.data, testset.targets
# import pdb;pdb.set_trace()
# classes = np.unique(train_label)
risk_score_list = [11.25, 2.25, 5.25, 5.0, 4.75,
                   8.0, 2.0, 11.0, 1.75, 10.75]

train_risk_scores = torch.tensor([risk_score_list[label] for label in train_label]).view(-1,1)
val_risk_scores = torch.tensor([risk_score_list[label] for label in val_label]).view(-1,1)
test_risk_scores = torch.tensor([risk_score_list[label] for label in test_label]).view(-1,1)


def x_to_gamma_dist(mean):
    '''
    creates gamma data, mean function of x
    '''

    var = 0.001
    alpha = mean.pow(2) / var
    beta = mean / var

    # alpha is shape
    # beta is rate, which is 1 / scale
    t = Gamma(alpha, beta).sample()
    qt = np.quantile(t, 0.9)
    c = Uniform(low=t.min(), high=qt).sample()

    # apply censoring
    observed_event = t <= c
    observed_time = torch.where(observed_event, t, c)

    return observed_time, observed_event.type(torch.int8)

train_time, train_event = x_to_gamma_dist(train_risk_scores)
val_time, val_event = x_to_gamma_dist(val_risk_scores)
test_time, test_event = x_to_gamma_dist(test_risk_scores)

train_labels = torch.cat([train_time, train_event],dim=1)
train_labels = train_labels.numpy()
val_labels = torch.cat([val_time, val_event],dim=1)
val_labels = val_labels.numpy()
test_labels = torch.cat([test_time, test_event],dim=1)
test_labels = test_labels.numpy()

train_df = pd.DataFrame(train_labels, columns=["time","dead"])
train_df["digit"] = train_label.numpy()
train_df["risk_score"] = train_risk_scores.numpy()

val_df = pd.DataFrame(val_labels, columns=["time","dead"])
val_df["digit"] = val_label.numpy()
val_df["risk_score"] = val_risk_scores.numpy()

test_df = pd.DataFrame(test_labels, columns=["time","dead"])
test_df["digit"] = test_label.numpy()
test_df["risk_score"] = test_risk_scores.numpy()
def normlize_data(train_data, val_data, test_data):
    train, valid, test = torch.tensor(train_data), torch.tensor(val_data), torch.tensor(test_data)

    #数据标准化
    min_val = train[train[:,-1] == 1.0,-2].min()
    max_val = train[train[:,-1] == 1.0,-2].max()
    deta_T = (max_val-min_val)/(Kbins-2.2)#时间精度，min_val对应0.1，max_val对应K-2.1
    min_val1 = max(min_val-0.1*deta_T,0)#缺失数据最小值，留margin，对应0
    max_val1 = min_val1+deta_T*(Kbins-1.1)#缺失数据最大值，对应K-1
    max_val2 = min_val1+deta_T*Kbins#最后一个bin代表infinite，对应K
    print(min_val, min_val1)
    print(max_val,max_val1,max_val2)
    # import pdb;pdb.set_trace()
    train[:,-2] = torch.clamp(train[:,-2],min_val1,max_val1)
    train[:,-2] = (train[:,-2] - min_val1) / (max_val2 - min_val1)
    
    valid[:,-2] = torch.clamp(valid[:,-2],min_val1,max_val1)
    valid[:,-2] = (valid[:,-2] - min_val1) / (max_val2 - min_val1)
    
    test[:,-2] = torch.clamp(test[:,-2],min_val1,max_val1)
    test[:,-2] = (test[:,-2] - min_val1) / (max_val2 - min_val1)
    
    return train, valid, test


# import pdb;pdb.set_trace()
# torch.save(train_data,'./triple_loss/data/MNIST/mnist_train_data.pt')
# train_df.to_csv('./triple_loss/data/MNIST/mnist_train_survival.csv')

# torch.save(val_data,'./triple_loss/data/MNIST/mnist_valid_data.pt')
# val_df.to_csv('./triple_loss/data/MNIST/mnist_valid_survival.csv')

# torch.save(test_data,'./triple_loss/data/MNIST/mnist_test_data.pt')
# test_df.to_csv('./triple_loss/data/MNIST/mnist_test_survival.csv')

train_data = train_df[["time","dead"]]
train_data = train_data.to_numpy()
train_data = train_data.astype(np.float64) 

val_data = val_df[["time","dead"]]
val_data = val_data.to_numpy()
val_data = val_data.astype(np.float64) 

test_data = test_df[["time","dead"]]
test_data = test_data.to_numpy()
test_data = test_data.astype(np.float64) 
# import pdb;pdb.set_trace()
train, valid, test = normlize_data(train_data, val_data, test_data)
torch.save(train, './triple_loss/data/MNIST/mnist_train_survival.pt')
torch.save(valid, './triple_loss/data/MNIST/mnist_valid_survival.pt')
torch.save(test, './triple_loss/data/MNIST/mnist_test_survival.pt')




