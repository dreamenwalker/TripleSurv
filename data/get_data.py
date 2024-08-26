import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import numpy as np
torch.manual_seed(1)
eps = 1e-6
Kbins = 50
root_dir = os.path.dirname(os.path.abspath(__file__))
dataset = 'BIDDING'
event_name = "OS"
filename = 'cleaned_features_v1.csv'
endfix = filename.split(".")[0]

data_path = rf'{root_dir}/{dataset}/{filename}'
if dataset in ["NPC",'NPC-MRI']:
    data = pd.read_csv(data_path,index_col=0)
else:
    data = pd.read_csv(data_path)

if dataset == "NPC":
    ncol = data.shape[1]
    name = ["OS", "DMFS", "LRRFS", "DFS", "OS.time", "DMFS.time", "LRRFS.time", "DFS.time"]
    data1 = data.drop(labels=name,axis=1)
    data1["dead"] = data[event_name]
    time_name = event_name+".time"
    data1["time"] = data[time_name]
    data = data1
colname = data.columns.to_list()
print(colname)
#确保最后两列分别是time和dead
if colname[-1] != "dead":
    dead = data["dead"]
    data.drop(labels=['dead'], axis=1,inplace = True)
    data['dead'] = dead
df = data
data = data.to_numpy()
data = data.astype(np.float64) 
nr,nc = data.shape
#数据标准化
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
# 5折交叉验证
if dataset == "NPC":
    kf = KFold(n_splits=5, shuffle=True,random_state = 123)
else:
    kf = KFold(n_splits=5)#对于npc数据
for k, (train_index, test_index) in enumerate(kf.split(data)):
    train_val_data, test_data = data[train_index,:], data[test_index,:]
    # import pdb;pdb.set_trace()
    df_val_data,df_test = df.iloc[train_index,:], df.iloc[test_index,:]
    train_data, val_data = train_test_split(train_val_data,test_size = 0.2,random_state = 0)
    _, supp_data = train_test_split(train_data, test_size = 0.1,random_state = 0)
    val_data = np.concatenate([val_data,supp_data], axis = 0)
    # import pdb;pdb.set_trace()
    # for i in range(7):#nc-2
        # print("normlized the feature: {}".format(colname[i]))
        # mv = np.mean(train_data[:,i])
        # std = np.std(train_data[:,i])
        # # import pdb;pdb.set_trace()
        # train_data[:,i] = (train_data[:,i]-mv)/(std+eps)
        # val_data[:,i] = (val_data[:,i]-mv)/(std+eps)
        # test_data[:,i] = (test_data[:,i]-mv)/(std+eps)
    # import pdb;pdb.set_trace()
    # train, valid, test = normlize_data(train_val_data, val_data, test_data)
    # torch.save(train, rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_tr.pt')
    # torch.save(test, rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_va.pt')
    # torch.save(test, rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_te.pt')
    train_data = pd.DataFrame(train_data, columns=colname)
    val_data = pd.DataFrame(val_data, columns=colname)
    train_data.to_csv(rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_tr.csv', index=True)
    val_data.to_csv(rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_val.csv', index=True)
    df_test.to_csv(rf'{root_dir}/{dataset}/{endfix}_{event_name}_{k}_te.csv', index=True)
    # import pdb;pdb.set_trace()
