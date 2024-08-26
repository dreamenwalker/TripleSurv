import torch
from sksurv.metrics import integrated_brier_score
import numpy as np
import pandas as pd
def get_integrated_brier_score(train_y,test_y,pred_values,step=1,end=1):
    '''

    Parameters
    ----------
    train_y : numpy array, (surv_time, surv_svent)
        DESCRIPTION.
    test_y : numpy array, (surv_time, surv_svent)
        DESCRIPTION.
    pred_values : numpy array
        DESCRIPTION.
    step : TYPE, optional
        DESCRIPTION. The default is 1.
    end : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    student_type = {'names': ('dead', 'time'), 'formats': ('bool', 'f8')}
    y1 = [(bool(train_y[i,1]),train_y[i,0]) for i in range(train_y.shape[0])]
    y2 = [(bool(test_y[i,1]),test_y[i,0]) for i in range(test_y.shape[0])]
    y1 = np.array(y1,dtype=student_type)
    y2 = np.array(y2,dtype=student_type)
    num_Category = pred_values.shape[1]
    surv_prob = 1-np.cumsum(pred_values,axis=1)
    times = []
    surv_preds = []
    for i in range(step-1,num_Category-end,step):
        times.append((0.5+i)/num_Category)
        surv_preds.append(surv_prob[:,i])
    surv_preds = np.stack(surv_preds,axis=1)
    times = np.array(times)
    
    return integrated_brier_score(y1,y2,surv_preds,times)

def myfun():
    wb = ["/home/zlz/Pyradiomics/triple_loss/data/BIDDING/cleaned_features_v1_0_tr.pt",
        "/home/zlz/Pyradiomics/triple_loss/model_output/bidding/deep_mtlr/sgd+likelihood0.5_SelfRank_cal5/0_model_outputs.csv",
        "/home/zlz/Pyradiomics/triple_loss/model_output/bidding/deep_mtlr/sgd+likelihood0.5_HitRank/0_model_outputs.csv"]
    
    model_name = []
    train_data = torch.load(wb[0])
    ncol = train_data.shape[-1]
    idx = (ncol-2, ncol-1)
    train_y = data[:, idx].float()
    train_y = train_y.numpy()
    res = []
    for i in range(1,len(wb)):
        data = pd.read_csv(wb[i])
        surv_e = data["dead"].to_numpy()
        surv_t = data["time"].to_numpy()
        tgt = data["pred_risk"].to_numpy()

        data_array = data.to_numpy()
        pred_values = data_array[:,:50]
        ncol = data_array.shape[-1]
        idx = (ncol-2, ncol-1)
        test_y = data_array[:,idx]
        ibs = get_integrated_brier_score(train_y,test_y,pred_values,step=2)
        res.append(ibs)