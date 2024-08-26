import torch
import torch.nn as nn
eps = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XCAL(nn.Module):
    def __init__(self, num_dcal_bins, train_gamma):
        super(XCAL, self).__init__()
        self.num_dcal_bins = num_dcal_bins
        self.train_gamma = train_gamma

    def get_cdf(self, pred_params, times):
        '''
        :param pred_params: 网络输出（没经过softmax）
        :param times: 对应到bin后的times
        :return:
        '''
        times = times.long()
        batch_sz = pred_params.size()[0]
        K = pred_params.size()[-1]
        times = times.view(-1, 1)
        indices = torch.arange(K).view(1, -1).to(DEVICE)

        # compute some masks
        # 1's up to but not including correct bin, then zeros
        mask1 = (times > indices).float()
        # 1 up to and including correct bin, then zeros
        mask2 = (times >= indices).float()

        cdf_km1 = (pred_params * mask1).sum(dim=-1)
        prob_k = pred_params[range(batch_sz), times.squeeze()]

        cdf_k = (pred_params * mask2).sum(dim=-1)
        assert torch.all((cdf_k - (cdf_km1 + prob_k)).abs() < 1e-4)
        
        return cdf_k

    def d_calibration(self, points, is_alive, nbins=20, differentiable=False, gamma=1.0, device='cpu'):
        # each "point" in points is a time for datapoint i mapped through the model CDF
        # each such time_i is a survival time if not censored or a time sampled
        # uniformly in (censor time, max time
        # compute empirical cdf of cdf-mapped-times
        # Move censored points with cdf values greater than 1 - 1e-4 t0 uncensored group
        new_is_alive = is_alive.detach().clone()
        new_is_alive[points > 1. - 1e-4] = 0

        points = points.to(device).view(-1, 1)
        # print(points[:200])
        # BIN DEFNITIONS
        # BIN DEFNITIONS
        # BIN DEFNITIONS
        # BIN DEFNITIONS
        # BIN DEFNITIONS
        # BIN DEFNITIONS
        bin_width = 1.0 / nbins
        bin_indices = torch.arange(nbins).view(1, -1).float().to(device)
        # print(bin_indices.shape)
        bin_a = bin_indices * bin_width  # + 0.02*torch.rand(size=bin_indices.shape)
        # print(bin_a.shape)
        noise = 1e-6 / nbins * torch.rand(size=bin_indices.shape).to(device)
        if not differentiable:
            noise = noise * 0.
        cum_noise = torch.cumsum(noise, dim=1)
        bin_width = torch.tensor([bin_width] * nbins).to(device) + cum_noise
        bin_b = bin_a + bin_width
        # print(bin_b.shape)

        bin_b_max = bin_b[:, -1]
        bin_b = bin_b / bin_b_max
        bin_a[:, 1:] = bin_b[:, :-1]
        bin_width = bin_b - bin_a

        # CENSORED POINTS
        points_cens = points[new_is_alive.long() == 1]
        points_cens = points_cens.view(-1, 1)
        # print("***", points.shape)
        # print(points_cens.shape,bin_b.shape)
        upper_diff_for_soft_cens = bin_b - points_cens
        # To solve optimization issue, we change the first left bin boundary to be -1.;
        # we change the last right bin boundary to be 2.
        bin_b[:, -1] = 2.
        bin_a[:, 0] = -1.
        lower_diff_cens = points_cens - bin_a  # p - a
        upper_diff_cens = bin_b - points_cens  # b - p
        diff_product_cens = lower_diff_cens * upper_diff_cens
        # NON-CENSORED POINTS

        if differentiable:
            # sigmoid(gamma*(p-a)*(b-p))
            bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
            exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)
        else:
            # (p-a)*(b-p)
            bin_index_ohe = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
            exact_bins_next = (lower_diff_cens <= 0).float()  # all bins after correct bin

        EPS = 1e-13
        right_censored_interval_size = 1 - points_cens + EPS

        # each point's distance from its bin's upper limit
        upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

        # assigns weights to each full bin that is larger than the point
        # full_bin_assigned_weight = exact_bins*bin_width
        # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
        full_bin_assigned_weight = (
                    exact_bins_next * bin_width.view(1, -1) / right_censored_interval_size.view(-1, 1)).sum(0)
        partial_bin_assigned_weight = (upper_diff_within_bin / right_censored_interval_size).sum(0)
        assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (
        full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

        # NON-CENSORED POINTS
        # NON-CENSORED POINTS
        # NON-CENSORED POINTS
        # NON-CENSORED POINTS
        # NON-CENSORED POINTS
        # NON-CENSORED POINTS
        points_uncens = points[new_is_alive.long() == 0]
        points_uncens = points_uncens.view(-1, 1)
        # compute p - a and b - p
        lower_diff = points_uncens - bin_a
        upper_diff = bin_b - points_uncens
        diff_product = lower_diff * upper_diff
        assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
        assert lower_diff.shape == (points_uncens.shape[0], bin_a.shape[1])
        # NON-CENSORED POINTS

        if differentiable:
            # sigmoid(gamma*(p-a)*(b-p))
            soft_membership = torch.sigmoid(gamma * diff_product)
            fraction_in_bins = soft_membership.sum(0)
            # print('soft_membership', soft_membership)
        else:
            # (p-a)*(b-p)
            exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
            fraction_in_bins = exact_membership.sum(0)

        assert fraction_in_bins.shape == (nbins,), fraction_in_bins.shape

        frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) / points.shape[0]
        
        return torch.pow(frac_in_bins - bin_width, 2).sum()
        
    def forward(self, pred_params, times, dead):
        '''
        :param pred_params: N * K 输出概率
        :param times: N, 离散化时间对应到bin
        :param dead: N, 0代表生存，1代表死亡
        :return:
        '''
        is_alive = 1.0 - dead
        cdf = self.get_cdf(pred_params, times)
        # print("cdf", cdf.shape)
        d_cal = self.d_calibration(points=cdf,
                                   is_alive=is_alive,
                                   nbins=self.num_dcal_bins,
                                   differentiable=True,
                                   gamma=self.train_gamma,
                                   device=DEVICE)
        return d_cal

class crps(nn.Module):
    def __init__(self, K):
        super(crps, self).__init__()
        self.K = K
        self.bin_boundaries = torch.linspace(0, 1.0, steps=K+1)
    def forward(self, pred_params, times, dead):
        '''

        :param pred_params: N * K 输出概率
        :param times: N, 离散化时间对应到bin
        :param dead: N, 0代表生存，1代表死亡
        :return:
        '''
        is_alive = 1.0 - dead
        times = times.to(DEVICE)
        times = times.view(-1, 1)

        bin_boundaries = self.bin_boundaries.to(DEVICE)
        bin_len = bin_boundaries[1:] - bin_boundaries[:-1]
        bin_len = bin_len.unsqueeze(-1)
        is_alive = is_alive.to(DEVICE)

        cdf = torch.cumsum(pred_params, dim=-1)
        indices = torch.arange(pred_params.shape[1]).view(
            1, -1).to(DEVICE)  # 只能用于将一个张量变形为一行或一列
        mask = (times >= indices).float()
        # SCRPS RIGHT
        loss =  torch.mm(mask * (cdf**2), bin_len)\
                + (1-is_alive) * torch.mm((1-mask)*((1 - cdf) ** 2), bin_len)
        loss = torch.mean(loss)
        # print(loss.shape)
        return loss
        
def likelihood_loss(pred_value, surv_mask1):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
        surv_mask1: batch_num * K | torch.tensor
    Output:
        loss: scalar
    ''' 
    R = pred_value*surv_mask1
    logp = torch.sum(R, dim=-1).log()
    loss = -1 * torch.mean(logp)

    return loss

def partial_likelihood_loss_breslow(risk, Y_label_T, Y_label_E):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
    Output:
        rank_loss: scalar
    ''' 
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    # Y_label_E = (Y_c > 0).float()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T,dtype=torch.float)
    mat_C = ((torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)) >= 0).float()#mat_C(i,j) = (tj-ti)>=0
    exp_H = torch.exp(risk)
    R = mat_C * (exp_H.T)# Get the risk in risk set
    R = torch.sum(R, dim=-1, keepdim=True)#batch_num * 1
    R = R + eps

    suma = torch.sum(Y_label_E*(risk - torch.log(R)))
    num = torch.sum(Y_label_E)

    paritial_loss = -1 * suma / (num + eps)

    return paritial_loss

def partial_likelihood_loss_efron(risk, Y_label_T, Y_label_E):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
    Output:
        rank_loss: scalar
    ''' 
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    # Y_label_E = (Y_c > 0).float()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T,dtype=torch.float)
    mat_A = ((torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)) > 0).float()#mat_A(i,j) = (tj-ti)>0
    mat_B = ((torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)) == 0).float()#mat_B(i,j) = (tj-ti)==0
    for i in range(B):
        mat_B[i,i+1:] = 0
    exp_H = torch.exp(risk)
    mat_C = ((mat_A+mat_B)>0).float()
    R = (mat_C) * (exp_H.T)
    R = torch.sum(R, dim=-1, keepdim=True)#batch_num * 1
    R = R + eps

    suma = torch.sum(Y_label_E*(risk - torch.log(R)))
    num = torch.sum(Y_label_E)

    paritial_loss = -1 * suma / (num + eps)

    return paritial_loss

def self_ranking_loss_weight(risk, Y_label_T, Y_label_E, sigma, scale):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
        sigma, scale: scalar
    Output:
        rank_loss: scalar
    '''
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    # Y_label_E = (Y_c > 0).float()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T, dtype=torch.float)

    R = torch.matmul(risk, one_vector.T) - torch.matmul(one_vector, risk.T)#R(i,j) = risk(i) - risk(j)
    T = torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)# T(i,j) = y(j) - y(i)
    mat_C = (T.sign() + 1 - Y_label_E.T).sign().relu()
    I_3 = Y_label_E*mat_C# I_3(i,j) = sign(yj - yi) + 1 - ej

    num = torch.sum(I_3)
    suma =  torch.sum(I_3 * torch.exp(sigma*(scale*T - R)))#sigma*(scale*T(i,j) - R(i,j))
    rank_loss = suma / (num + eps)

    return rank_loss
def ranking_loss_hit(preds, Y_label_T, Y_label_E, mask2, sigma):
    '''
    Input:
        preds: batch_num * K | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
        mask2: batch_num * K | torch.tensor
        sigma, scale: scalar
    Output:
        rank_loss: scalar
    '''

    B,L = preds.shape
    I_2 = Y_label_E.squeeze().diag()
    one_vector = torch.ones_like(Y_label_T, dtype=torch.float)

    R = torch.matmul(mask2, preds.T)#F(ti,xj)
    diag_R = torch.reshape(torch.diag(R), [-1, 1])
    R = diag_R - R# R(i,j) = F(ti,xi) - F(ti,xj)
    T = torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)# T(i,j) = y(j) - y(i)
    I_3 = torch.matmul(I_2, T.sign().relu())# I_3(i,j) = e(i) * (y(j) - y(i))
    
    num = torch.sum(I_3)
    suma =  torch.sum(I_3 * torch.exp(-sigma*R))#-sigma*R(i,j)
    rank_loss = suma / (num + eps)

    return rank_loss
                      
def ranking_loss(risk, Y_label_T, Y_label_E, sigma):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
        sigma, scale: scalar
    Output:
        rank_loss: scalar
    '''
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    I_2 = Y_label_E.squeeze().diag()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T, dtype=torch.float)

    R = torch.matmul(risk, one_vector.T) - torch.matmul(one_vector, risk.T)#R(i,j) = risk(i) - risk(j)
    T = torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)# T(i,j) = y(j) - y(i)
    I_3 = torch.matmul(I_2, T.sign().relu())# I_3(i,j) = e(i) * (y(j) - y(i))

    num = torch.sum(I_3)
    suma =  torch.sum(I_3 * torch.exp(-sigma*R))#-sigma*R(i,j)
    rank_loss = suma / (num + eps)

    return rank_loss

def calibration_loss(pred_value, Y_label_E, surv_mask1, surv_mask2, nbins):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        surv_mask1: batch_num * K
        surv_mask2: batch_num * K
        nbins: scalar
    Output:
        rank_loss: scalar
    '''

    num_time = pred_value.shape[1]
    I_2 = Y_label_E * surv_mask1

    new_mask1 = []
    new_mask2 = []
    pred_p = []
    time_step = int(num_time / nbins)

    for i in range(nbins):
        A = surv_mask2[:, i * time_step : (i + 1) * time_step]
        A = torch.max(A, dim = -1, keepdim = True).values
        B = I_2[:, i * time_step : (i + 1) * time_step]
        B = torch.max(B, dim = -1, keepdim = True).values
        C = pred_value[:, i * time_step : (i + 1) * time_step]
        C = torch.sum(C, dim = -1, keepdim=True)

        new_mask1.append(A)
        new_mask2.append(B)
        pred_p.append(C)

    new_mask1 = torch.cat(new_mask1, dim = -1)
    new_mask2 = torch.cat(new_mask2, dim = -1)
    pred_p = torch.cat(pred_p, dim = -1)
    pred_s = 1.0 + pred_p - torch.cumsum(pred_p, dim = -1)#S = 1 + p - cumsum(p)
    pred_h = pred_p / (eps+pred_s)
    r = torch.mean(pred_h, dim = 0)  # no need to divide by each individual dominator
    t_a = torch.sum(new_mask1, dim = 0)
    t_b = torch.sum(new_mask2, dim = 0)
    t = t_b / (eps + t_a)

    cal_loss = torch.mean((r - t) ** 2)

    return cal_loss