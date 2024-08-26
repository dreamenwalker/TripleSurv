import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .backbone.transformer.vit.vit_ori import Transformer

def Get_act(name):
    if name == "relu":
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "tanh":
        act = nn.Tanh()
    elif name == "softmax":
        act = nn.Softmax(dim=-1)
    elif name == "sigmoid":
        act = nn.Sigmoid()
    else:
        raise Exception("No implementation of the activation for {}".format(name))
    return act

class DeepNet(nn.Module):
    def __init__(self, D_in, HS, dropout_rate, act_layer='relu', batchnorm=True):
        super(DeepNet, self).__init__()
        # print("#################################dropout_rate: {}".format(dropout_rate))
        self.fc1 = nn.Linear(D_in, HS[0])
        self.drop1 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(HS[0])
        self.fc2 = nn.Linear(HS[0], HS[1])
        self.drop2 = nn.Dropout(dropout_rate)
        self.act = Get_act(act_layer)

        self.bn2 = nn.BatchNorm1d(HS[1] + D_in)
        self.fc3 = nn.Linear(HS[1] + D_in, HS[2])
        self.drop3 = nn.Dropout(dropout_rate)
        self.batchnorm = batchnorm

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        res = x
        if self.batchnorm:
            x = self.drop1(self.act(self.fc1(x)))
            x = self.drop2(self.act(self.fc2(self.bn1(x))))
            x = torch.cat((x, res), dim=-1)
            x = self.drop3(self.act(self.fc3(self.bn2(x))))
        else:
            x = self.drop1(self.act(self.fc1(x)))
            x = self.drop2(self.act(self.fc2(x)))
            x = torch.cat((x, res), dim=-1)
            x = self.drop3(self.act(self.fc3(x)))

        return x

class MLP(nn.Module):
    def __init__(self, D_in, HS, act_layer='relu', out_layer='relu', batchnorm=True):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(D_in, HS[0])

        self.bn1 = nn.BatchNorm1d(HS[0])
        self.fc2 = nn.Linear(HS[0], HS[1])

        self.act = Get_act(act_layer)
        self.out_act = Get_act(out_layer)
        

        self.bn2 = nn.BatchNorm1d(HS[1] + D_in)
        self.fc3 = nn.Linear(HS[1] + D_in, HS[2])
        self.batchnorm = batchnorm

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        if self.batchnorm:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(self.bn1(x)))
            x = self.out_act(self.fc3(self.bn2(x)))
        else:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.out_act(self.fc3(x))

        return x

class Model_DeepHit(nn.Module):
    def __init__(self, D_in, num_cat_bins=50, hidden_sizes = [32, 32, 64], act_layer='relu',dropout_rate=0.2, out_layer='softmax', **kwargs):
        super(Model_DeepHit, self).__init__()
        self.backbone = DeepNet(D_in, hidden_sizes, dropout_rate, act_layer=act_layer)
        self.backbone.apply(self.backbone.init_weights)
        self.head = nn.Linear(hidden_sizes[-1], num_cat_bins)
        self.out_act = Get_act(out_layer)

    def forward(self, src):
        src = self.backbone(src)
        pred = self.out_act(self.head(src))
        return pred

class Model_DeepCox(nn.Module):
    def __init__(self, D_in, num_cat_bins=1, hidden_sizes = [32, 32, 64], act_layer='relu',dropout_rate=0.2, out_layer='tanh', **kwargs):
        super(Model_DeepCox, self).__init__()
        self.backbone = DeepNet(D_in, hidden_sizes, dropout_rate, act_layer=act_layer)
        self.backbone.apply(self.backbone.init_weights)
        self.head = nn.Linear(hidden_sizes[-1], 1)
        self.out_act = Get_act(out_layer)

    def forward(self, src):
        src = self.backbone(src)
        pred = self.out_act(self.head(src))

        return pred

class Model_DeepCox_Hazard(nn.Module):
    def __init__(self, D_in, num_cat_bins=50, hidden_sizes = [32, 32, 64], act_layer='relu',dropout_rate=0.2, out_layer='tanh',**kwargs):
        super(Model_DeepCox_Hazard, self).__init__()
        self.backbone = DeepNet(D_in, hidden_sizes, dropout_rate, act_layer=act_layer)
        self.backbone.apply(self.backbone.init_weights)
        self.head = nn.Linear(hidden_sizes[-1], num_cat_bins)
        self.out_act = Get_act(out_layer)

        self.base_risk = nn.parameter.Parameter(torch.zeros(1, num_cat_bins))
        self.risk_model = MLP(num_cat_bins, hidden_sizes=[num_cat_bins, num_cat_bins, num_cat_bins], act_layer=act_layer, out_layer=out_layer)
        self.risk_model.apply(self.risk_model.init_weights)

    def forward(self, src):
        src = self.backbone(src)
        prog_score = self.out_act(self.head(src))
        out = torch.exp(prog_score) * self.risk_model(self.base_risk)
        # out = torch.sigmoid(torch.exp(torch.tanh(prog_score)) * torch.relu(self.base_risk))  # [h1, h2, ...]
        # out = torch.sigmoid(out)
        # out = torch.softmax(out,dim=-1)
        T = torch.zeros_like(out).float()
        T[:, 1:] = 1 - out[:, :-1]
        T[:, 0] = 1.  # [1., 1 - h1, 1- h2,...]
        T = torch.cumprod(T, dim=-1)  # [1., 1-h1, (1-h1)*(1-h2), ...]
        pred = out * T  # [h1, h2*(1-h1), h3*(1-h1)*(1-h2)
        if torch.isnan(pred.sum()):
            import pdb;
            pdb.set_trace()

        return pred, prog_score

class Model_MTLR(nn.Module):
    def __init__(self, D_in, num_cat_bins=50, hidden_sizes = [32, 32, 64], batchnorm=True, act_layer='relu',dropout_rate=0.2,out_layer="softmax",**kwargs):
        super(Model_MTLR, self).__init__()
        self.backbone = DeepNet(D_in, hidden_sizes, dropout_rate, act_layer=act_layer, batchnorm=batchnorm)
        self.backbone.apply(self.backbone.init_weights)
        self.head = nn.Linear(hidden_sizes[-1], num_cat_bins-1)
        self.out_act = Get_act(out_layer)
        
    def forward(self, src):
        out = self.backbone(src)
        out = self.head(out)
        out_concat = torch.cat([torch.zeros_like(out[:, :1]), out], dim=-1)
        out = torch.sum(out, dim=-1, keepdim=True) - torch.cumsum(out_concat, dim=-1)
        if torch.isnan(out).any():
            import pdb;pdb.set_trace()
        out = self.out_act(out)

        return out

class Model_DeepCox_TF(nn.Module):
    def __init__(self, embed_dim, class_flag=False, feat_embded=False, depth=12, num_heads=12, dropout_rate=0.2, mlp_ratio=3, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super(Model_DeepCox, self).__init__()
        self.backbone = DeepNet(D_in, hidden_sizes, dropout_rate, act_layer=act_layer)
        self.backbone.apply(self.backbone.init_weights)
        self.head = nn.Linear(hidden_sizes[-1], 1)
        self.out_act = Get_act(out_layer)

    def forward(self, src):
        src = self.backbone(src)
        pred = self.out_act(self.head(src))

        return pred
