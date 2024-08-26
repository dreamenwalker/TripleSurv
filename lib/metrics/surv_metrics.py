# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 22:43:57 2022

@author: 18292
"""
#全部用R来实现
# from lifelines.utils import concordance_index
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 
from scipy.stats import chi2
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter

# from lifelines import KaplanMeierFitter
# from lifelines.datasets import load_waltons
# waltons = load_waltons()

# kmf = KaplanMeierFitter(label="waltons_data")
# kmf.fit(waltons['T'], waltons['E'])
# x = kmf.survival_function_
# kmf.survival_function_at_times(30).to_numpy()[0]


def KM_estimator(surv_e, surv_t, tgt):
    kmf = KaplanMeierFitter(label="survival data")
    kmf.fit(surv_t, surv_e)
    return kmf.survival_function_at_times(tgt).to_numpy()[0]

### BRIER-SCORE for a fully uncensored dataset
def brier_score(pred_values, surv_time, surv_event, mid_points, at_time):
    risk_cumsum = np.cumsum(pred_values,axis=1)
    risk_at_time = 1.0-risk_cumsum[:,at_time]
    tgt = mid_points[at_time]
    
    y_true = ((surv_time <= tgt) * surv_event).astype(float)

    return np.mean((risk_at_time - y_true)**2)

def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G

# this account for the weighted average for unbaised estimation
def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0,:] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0,:] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i])*float(Y_test[i])/G1 + Y_tilde[i]/G2

    # y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W*(Y_tilde - (1.-Prediction))**2)

def AUROC(surv_event, prog_risk, main='ROC curve (area = %0.2f)'):
    fpr,tpr,threshold = roc_curve(surv_event, prog_risk) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
     
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=main % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

def TimeDependent_AUROC(surv_time, surv_event, pred_values,at_time,mid_points):
    risk_cumsum = np.cumsum(pred_values,axis=1)
    risk_at_time = risk_cumsum[:,at_time]
    tgt = mid_points[at_time]
    mask =  (surv_event!=0.0) | (surv_time>=tgt)
    new_time = surv_time[mask]
    target_risk = risk_at_time[mask]
    target_event = (new_time<tgt).astype(np.int)
    
    return AUROC(target_event, target_risk, main=str(tgt)+'-time ROC curve (area = %0.2f)')
   

def calibartion_curve(surv_time, surv_event, pred_values, at_time, mid_points,k=3):
    surv_time = surv_time.squeeze()
    surv_event = surv_event.squeeze()
    mid_points = mid_points.squeeze()
    #get F(t*) = 1-S(t*)
    risk_cumsum = np.cumsum(pred_values,axis=1)
    risk_at_time = risk_cumsum[:,at_time]
    tgt = mid_points[at_time]
    #sort S(.) into k groups defined by quantiles
    sort_ind = np.argsort(risk_at_time)
    sort_risk = risk_at_time[sort_ind]
    sort_time = surv_time[sort_ind]
    sort_event = surv_event[sort_ind]
    inds = np.linspace(0, len(risk_at_time), num=k+1)
    test_statistic = 0
    bin_a = 0
    observed_data = []
    predicted_data = []
    for i in range(k):
        bin_b = round(inds[i+1])
        risk = sort_risk[bin_a:bin_b]
        surv_t = sort_time[bin_a:bin_b]
        surv_e = sort_event[bin_a:bin_b]
        num_g = bin_b-bin_a
        num_eprisk = np.sum(risk)
        mean_erisk = num_eprisk/num_g
        num_obrisk = num_g*(1-KM_estimator(surv_e,surv_t,tgt))
        test_statistic += (num_obrisk-num_eprisk)**2/(num_g*mean_erisk(1-mean_erisk))
        observed_data.append(num_obrisk/num_g)
        predicted_data.append(num_eprisk/num_g)
    #chi-square test
    rv = chi2(k-1)
    # p value = 1-rv.cdf(test_statistic)
    p_value = 1. - rv.cdf(test_statistic)
    
    #plot calibration curve
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(num_obrisk, num_eprisk, color='darkorange',
             lw=lw, label="HL test: %.2f" % p_value) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('observed rate of event')
    plt.ylabel('predicted rate of event')
    plt.title('Calibration curve at %.1f' % tgt)
    plt.legend(loc="lower right")
    plt.show()
    return p_value,observed_data,predicted_data
