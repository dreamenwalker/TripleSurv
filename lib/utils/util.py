import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_surv_mask1(surv_time, dead, num_Category = 20):
    '''
    Input:
        surv_time: batch_num * 1 or 1 * batch_num or batch_num | torch.tensor
        is_alive: batch_num * 1 or or 1 * batch_num or batch_num | 0 (death) 1 (censored) | torch.tensor
        num_Category: int
    Output:
        mask1: batch_num * num_Category | torch.tensor
    '''
    cat_time = get_cat_time(surv_time, num_Category = num_Category)
    cat_time = cat_time.view(-1)
    dead = dead.view(-1)

    mask1 = torch.zeros(cat_time.shape[0], num_Category)
    for i in range(mask1.shape[0]):
        if dead[i] == 1:
            mask1[i, int(cat_time[i])] = 1
        else:
            mask1[i, int(cat_time[i] + 1):] = 1

    mask1 = mask1.to(DEVICE)

    return mask1

def get_surv_mask2(surv_time, num_Category = 20):
    '''
    Input:
        surv_time: batch_num * 1 or or 1 * batch_num or batch_num | torch.tensor
        num_Category: int
    Output:
        mask1: batch_num * num_Category | torch.tensor
    '''
    cat_time = get_cat_time(surv_time, num_Category = num_Category)
    cat_time = cat_time.view(-1)

    mask2 = torch.zeros(cat_time.shape[0], num_Category)
    for i in range(mask2.shape[0]):
        t = int(cat_time[i]) + 1
        mask2[i, :t] = 1

    mask2 = mask2.to(DEVICE)

    return mask2

def get_risk(pred_value):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
    Output:
        risk: batch_num * 1
    '''
    K = pred_value.shape[1]
    time_line = 0.5*(torch.arange(1,K+1)+torch.arange(0,K))/K
    time_line = time_line.to(DEVICE)
    pred_value = pred_value * time_line
    risk = 1.0 - torch.sum(pred_value, dim=-1, keepdim=True)
    
    return risk


def get_cat_time(surv_time, num_Category = 20):

    bin_boundaries = torch.arange(1, num_Category+1)/num_Category
    bin_boundaries = bin_boundaries.view(1, -1)
    bin_boundaries = bin_boundaries.to(DEVICE)
    tte_cat = (surv_time >= bin_boundaries).sum(dim=-1)

    return tte_cat