import telnetlib
from traceback import print_tb
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter

class Pos_DROLoss(nn.Module):
    def __init__(self, temperature=1., temperature2=1., temperature3=1., loss_re=False, mode="test", device=None):
        super(Pos_DROLoss, self).__init__()
        self.temperature = temperature
        self.temperature_2 = temperature2
        self.temperature_3 = temperature3
        self.device = device
        self.loss_re = loss_re
        self.mode = mode
        print("Here is Pos_DROLoss loss")
        print("tau_1\ttau_2 tau_3 \t is {}\t{}\t {} loss_re {} MODE {}".format(temperature, temperature2, temperature3, self.loss_re, self.mode))

    def forward(self, y_pred, user):
        pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
        neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        # print(self.loss_re)
        if self.mode == "multi":
            user = user.contiguous().view(-1, 1)
            mask = torch.eq(user, user.T).float().to(self.device)
            pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
            neg_logits = torch.pow(torch.mean(neg_logits, dim=-1), self.temperature_2)
        elif self.mode == "single":
            pos_logits = pos_logits
            neg_logits = torch.mean(neg_logits, dim=-1)
        elif self.mode == "reweight":
            neg_logits = neg_logits.sum(dim=-1)
            neg_logits = torch.pow(neg_logits, self.temperature_2)
        elif self.mode == "once":
            unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
            pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
            neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
            neg_logits = torch.pow(neg_logits, self.temperature_2)
        if self.loss_re:
            # print("HERE")
            loss = - self.temperature_3 * torch.log(pos_logits / neg_logits).mean()
        else:
            loss = - torch.log(pos_logits / neg_logits).mean()
        return loss, None
