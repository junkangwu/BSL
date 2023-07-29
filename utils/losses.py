import telnetlib
from traceback import print_tb
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
class EasyContrastiveLoss(nn.Module):
    def __init__(self, negative_weight=None):
        super(EasyContrastiveLoss, self).__init__()
        self._negative_weight = negative_weight

    def forward(self, y_pred):
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits + 1)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        
        return loss.mean()

class BatchContrastiveLoss(nn.Module):
    def __init__(self, negative_weight=None):
        super(BatchContrastiveLoss, self).__init__()
        self._negative_weight = negative_weight
    
    def forward(self, y_pred, mask):
        batch_size = mask.shape[0]
        pos_logits = y_pred[~mask]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[mask].view(batch_size, -1)
        neg_loss = torch.relu(neg_logits + 1)

        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)

        return loss.mean()
        

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight
        print("Here is CCL loss, margin {} negative_weight {}".format(self._margin, self._negative_weight))
    def forward(self, y_pred):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        # pos_logits = y_pred[:, 0]
        # pos_loss = torch.pow(pos_logits - 1, 2) / 2
        # neg_logits = y_pred[:, 1:]
        # neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2

        pos_logits = y_pred[:, 0]
        neg_logits = y_pred[:, 1:]
        pos_loss = torch.relu(1 - pos_logits)
        neg_loss = torch.relu(neg_logits).sum(dim=-1)
        loss = self._negative_weight * pos_loss + neg_loss
        return loss.mean()
        # pos_logits = y_pred[:, 0]
        # pos_loss = torch.relu(1 - pos_logits)
        # neg_logits = y_pred[:, 1:]
        # neg_loss = torch.relu(neg_logits - self._margin)
        # # mask = torch.gt(neg_logits, self._margin)
        # # print("Valid value rate is {:.5}".format((mask.sum() / (mask.size(0) * mask.size(1))).item()))
        # if self._negative_weight:
        #     loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        # else:
        #     loss = pos_loss + neg_loss.sum(dim=-1)
        # return loss.mean()

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        print("Here is MSELoss")
    def forward(self, y_pred):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        # y_pred = y_pred.sigmoid()
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
        loss = pos_loss + neg_loss
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        print("Here is BCELoss")
    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        logits = y_pred.flatten()
        labels = y_true.flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        print("Here is BPRLoss")
    def forward(self, y_pred):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        logits_diff = pos_logits - neg_logits
        loss = -torch.log(torch.sigmoid(logits_diff)).mean()
        return loss

class MSEInforNCELoss(nn.Module):
    def __init__(self):
        super(MSEInforNCELoss, self).__init__()
    def forward(self, y_pred, mask):
        batch_size = y_pred.size(0)
        pos_logits = y_pred.masked_select(~mask)
        neg_logits = y_pred.masked_select(mask).view(batch_size, -1)

        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_loss = torch.pow()
        logits_diff = pos_logits - neg_logits
        loss = - torch.log(torch.sigmoid(logits_diff)).mean()
        return loss

class MSEWeightLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        super(MSEWeightLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(torch.relu(neg_logits - self._margin), 2).sum(dim=-1) / 2
        
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)

        return loss.mean()

class MSEWeightLoss2(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        super(MSEWeightLoss2, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
        

        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)

        return loss.mean()

class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # pos_logits = y_pred[:, 0].unsqueeze(-1)
        # neg_logits = y_pred[:, 1:]
        # logits_diff = pos_logits - neg_logits
        # loss = -torch.log(torch.sigmoid(logits_diff)).mean()

        pos_logits = torch.exp(y_pred[:, 0])
        neg_logits = torch.exp(y_pred[:, 1:])
        Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / Ng)).mean()
        return loss
class PairwiseLogisticEasy_2Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticEasy_2Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("margin is {}".format(self._margin))

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        neg_scores = y_pred[:, 1:]
        Ng = scatter(neg_logits[neg_scores > self._margin], torch.where(neg_scores > self._margin)[0], dim=0, reduce='sum', dim_size=neg_logits.size(0))
        # Ng = torch.
        loss = (- torch.log(pos_logits / (pos_logits + Ng)))
        # loss_1 = (- torch.log(pos_logits / Ng))
        # loss_2 = (- torch.log(pos_logits / neg_logits.sum(dim=-1)))

        # loss = torch.where(Ng==0., loss_2, loss_1)
        # print("loss min :{} max:{}".format(loss.min(), loss.max()))
        return loss, 0.

class PairwiseLogisticEasyLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticEasyLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("Here is Easy loss")
        print("tau is {}".format(temperature))

    def forward(self, y_pred, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.sum(dim=-1)
        # print("pos neg {} {:.4}".format(pos_logits, Ng))
        loss = (- torch.log(pos_logits / (Ng)))
        # loss = (- torch.log(pos_logits / Ng))

        return loss.mean()

class DORO_Loss(nn.Module):
    def __init__(self, temperature=1., choose_num=1.):
        super(DORO_Loss, self).__init__()
        self._temperature = temperature
        self._choose_num = int(choose_num)
        print("Here is DORO_Loss loss")
        print("tau tau is {} choose num is \t{}".format(temperature, self._choose_num))

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M

        rk = torch.argsort(neg_logits, descending=True)[:, self._choose_num:]
        Ng = torch.gather(neg_logits, 1, rk).sum(dim=-1)
        loss = (- torch.log(pos_logits / (Ng)))

        return loss.mean()

class DCL_Loss(nn.Module):
    def __init__(self, temperature=1., tau_plus=1., t3=1.):
        super(DCL_Loss, self).__init__()
        self._temperature = temperature
        self._tau_plus = tau_plus
        self.t_3 = t3
        print("Here is DCL loss")
        print("tau tau_plus is {}\t{}\t{}".format(temperature, tau_plus, self.t_3))

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        if self.t_3 > 1.: 
            Ng = ( - self._tau_plus * neg_logits.size(1) * pos_logits + neg_logits.sum(dim=-1)) / (1 - self._tau_plus)
        elif self.t_3 == 1:
            Ng = ( - self._tau_plus * neg_logits.size(1) * pos_logits.detach() + neg_logits.sum(dim=-1)) / (1 - self._tau_plus)
        else:
            Ng = ( - self._tau_plus * neg_logits.size(1) * pos_logits + neg_logits.sum(dim=-1))
        neg_rate = (Ng<=neg_logits.size(1) * np.e ** (-1 / self._temperature)).sum() / Ng.numel()
        Ng = torch.clamp(Ng, min=neg_logits.size(1) * np.e ** (-1 / self._temperature))
        # pos_rate = pos_logits / (neg_logits.sum(dim=-1) - )
        # Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / (Ng)))
        # loss = (- torch.log(pos_logits / Ng))

        return loss.mean(), neg_rate

class RinceLoss(nn.Module):
    def __init__(self, temperature=1., q=1., lam=1.):
        super(RinceLoss, self).__init__()
        self._temperature = temperature
        self._q = q
        self._lam = lam
        print("Here is RINCE loss")
        print("tau\tq\tlam is {}\t{}\t{}".format(temperature, q, lam))

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M
        Ng = neg_logits.sum(dim=-1)
        # RINCE loss
        neg = ((self._lam * ( Ng))**self._q) / self._q
        pos = - (pos_logits ** self._q) / self._q
        loss = pos.mean() + neg.mean()

        return loss

class DRO_reweightLoss(nn.Module):
    def __init__(self, temperature=1., q=1., lam=1.):
        super(DRO_reweightLoss, self).__init__()
        self._temperature = temperature
        self._q = q
        self._lam = lam
        print("Here is DRO_reweightLoss loss")
        print("tau\tq\tlam is {}\t{}\t{}".format(temperature, q, lam))

    def forward(self, y_pred):
        pos_logits = torch.exp(y_pred[:, 0] /  self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M
        Ng = neg_logits.sum(dim=-1)
        # RINCE loss
        Ng = torch.pow(Ng, self._q)
        loss = - torch.log( pos_logits / Ng).mean()

        return loss

class Pos_Reweight(nn.Module):
    def __init__(self, temperature=1., temperature2=1., temperature3=1., loss_re=False, mode="test", device=None):
        super(Pos_Reweight, self).__init__()
        self.temperature = temperature
        self.temperature_2 = temperature2
        self.temperature_3 = temperature3
        self.device = device
        self.loss_re = loss_re
        self.mode = mode
        print("Here is Pos_Reweight loss")
        print("tau_1\ttau_2 tau_3 \t is {}\t{}\t{}loss_re\t{} MODE {}".format(temperature, temperature2, temperature3, loss_re, self.mode))

    def forward(self, y_pred, user, t1, t2, t3, mode):
        pos_logits = torch.exp(y_pred[:, 0] / t1)
        neg_logits = torch.exp(y_pred[:, 1:] / t1)
        if mode == "multi":
            user = user.contiguous().view(-1, 1)
            mask = torch.eq(user, user.T).float().to(self.device)
            pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
            neg_logits = torch.pow(torch.mean(neg_logits, dim=-1), t2)
        elif mode == "single":
            pos_logits = pos_logits
            neg_logits = torch.mean(neg_logits, dim=-1)
        elif mode == "reweight":
            Ng = neg_logits.sum(dim=-1)
            pos_logits = torch.pow(pos_logits, t2)
            Ng = torch.pow(Ng, t3)
            loss = - torch.log(pos_logits / Ng).mean()
            return loss, None
        elif mode == "DCL":
            neg_logits = neg_logits.mean(dim=-1) # B M
            pos_logits = pos_logits.mean(dim=-1)  # [1]
            loss = torch.log(neg_logits).mean() - torch.log(pos_logits)
            return loss, None
        elif mode == "DCL_pre":
            neg_logits = neg_logits.mean(dim=-1) # B M
            pos_logits = pos_logits  # [1]
            print(pos_logits.size())
            loss = torch.log(neg_logits).mean() - torch.log(pos_logits).mean()
            return loss, None
        elif mode == "DCL_select":
            values, index = torch.topk(neg_logits, int(t3), dim=-1)
            pos_logits = torch.cat([pos_logits.unsqueeze(-1), values], dim=1).mean(dim=-1)
            neg_logits = neg_logits.mean(dim=-1) # B M
            # pos_logits = pos_logits  # [1]
            loss = torch.log(neg_logits).mean() - torch.log(pos_logits).mean()
            return loss, None
        elif mode == "once":
            unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
            pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
            neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
            neg_logits = torch.pow(neg_logits, self.temperature_2)
        if self.loss_re:
            loss = - self.temperature * torch.log(pos_logits / neg_logits).mean()
        else:
            loss = - torch.log(pos_logits / neg_logits).mean()
        return loss, None

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

class SSMLoss(nn.Module):
    def __init__(self, temperature=1., temperature2=1., temperature3=1., loss_re=False, mode="test", device=None):
        super(SSMLoss, self).__init__()
        self.temperature = temperature
        self.temperature_2 = temperature2
        self.temperature_3 = temperature3
        self.device = device
        self.loss_re = loss_re
        self.mode = mode
        print("Here is SSMLoss loss")
        print("tau_1\ttau_2 tau_3 \t is {}\t{}\tloss_re {} MODE {}".format(temperature, temperature2, temperature3, loss_re, self.mode))

    def forward(self, y_pred, user):
        pos_logits = torch.exp(y_pred[:, 0] / self.temperature_3)
        neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        if self.mode == "multi":
            user = user.contiguous().view(-1, 1)
            mask = torch.eq(user, user.T).float().to(self.device)
            pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
            neg_logits = torch.pow(torch.mean(neg_logits, dim=-1), self.temperature_2)
        elif self.mode == "single":
            pos_logits = pos_logits
            neg_logits = torch.mean(neg_logits, dim=-1)
        elif self.mode == "reweight":
            Ng = pos_logits + neg_logits.sum(dim=-1)
            Ng = torch.pow(Ng, self.temperature_2)
            # Ng = torch.clamp(Ng, min=neg_logits.size(1) * np.e ** (-1 / self.temperature))
            # RINCE loss
            loss = - torch.log(pos_logits / Ng).mean()
            return loss, None
        elif self.mode == "once":
            unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
            pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
            neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
            neg_logits = torch.pow(neg_logits, self.temperature_2)
        if self.loss_re:
            loss = - self.temperature * torch.log(pos_logits / neg_logits).mean()
        else:
            loss = - torch.log(pos_logits / neg_logits).mean()
        return loss, None

class R_CELoss(nn.Module):
    def __init__(self):
        super(R_CELoss, self).__init__()
        print("Here is R_CELoss loss")

    def forward(self, y, t, alpha):
        loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
        y_ = torch.sigmoid(y).detach()
        weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)
        loss_ = loss * weight
        loss_ = torch.mean(loss_)
        return loss_

class T_CELoss(nn.Module):
    def __init__(self):
        super(R_CELoss, self).__init__()
        print("Here is T_CELoss loss")

    def forward(self, y, t, drop_rate):
        loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
        loss_mul = loss * t
        ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

        return loss_update

class PairwiseLogisticTWOLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticTWOLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("Here is Easy loss")
        print("tau is {}".format(temperature))

    def forward(self, y_pred_pos, y_pred_neg, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred_pos / temperature_
        neg_logits = torch.log( torch.sum( torch.exp(y_pred_neg / temperature_), dim=-1))
        # loss = (- torch.log(pos_logits / Ng))
        loss = - pos_logits + neg_logits
        return loss

class InforNCEPosDROLoss(nn.Module):
    def __init__(self, temperature1=1., temperature2=1., eta=0.):
        super(InforNCEPosDROLoss, self).__init__()
        # self._margin = margin 
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.eta_ = eta

    def forward(self, y_pred, pos_num):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        pos_logits = torch.exp(y_pred[:, :1 + pos_num] / self.temperature1) # B
        neg_logits = torch.exp(y_pred[:, 1 + pos_num: ] / self.temperature2) # B
        # print(pos_logits.size())
        # print(neg_logits.size())
        pos_ = pos_logits.mean(dim=-1)
        Ng_ = torch.pow(neg_logits.mean(dim=-1), self.temperature1 / self.temperature2) * self.eta_
        # print(pos_)
        # print(neg_logits.mean())
        # print(self.temperature1 / self.temperature2)
        # print(self.eta_)
        # print(Ng_)
        loss = self.temperature1 * (- torch.log(pos_ / (Ng_) ))

        return loss
        
class PairwiseLogisticDROLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticDROLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("margin is {}".format(self._margin))

    def forward(self, y_pred, temperature_, eta_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.mean(dim=-1)
        # loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))
        loss = (- torch.log(pos_logits / (Ng * np.exp(eta_))))

        return loss

class PairwiseLogisticDRO2Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticDRO2Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("DRO22222 margin is {}".format(self._margin))

    def forward(self, y_pred, temperature_, eta_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.mean(dim=-1)
        # loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))
        # temperature_tmp = (temperature_ - temperature_.min()) / (temperature_.max() - temperature_.min())
        loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))

        return loss

class PairwiseLogisticDRO3Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticDRO3Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("DRO22222 margin is {}".format(self._margin))

    def forward(self, y_pred, temperature_, eta_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.mean(dim=-1)
        # loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))
        # temperature_tmp = (temperature_ - temperature_.min()) / (temperature_.max() - temperature_.min())
        loss = temperature_ * (- torch.log(pos_logits / (pos_logits + Ng * np.exp(eta_))))

        return loss

class BCEDROLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(BCEDROLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("BCEDROLoss margin is {}".format(self._margin))

    def forward(self, y_pred, temperature_, eta_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        y_pred = y_pred.sigmoid()
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.mean(dim=-1)
        # loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))
        # temperature_tmp = (temperature_ - temperature_.min()) / (temperature_.max() - temperature_.min())
        loss = temperature_ * (- torch.log(pos_logits / (pos_logits + Ng * np.exp(eta_))))

        return loss

class BCEDRO2Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(BCEDRO2Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("BCEDRO2Loss margin is {}".format(self._margin))

    def forward(self, y_pred, temperature_, eta_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        # alpha
        y_pred = y_pred.sigmoid()
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.mean(dim=-1)
        # loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))
        # temperature_tmp = (temperature_ - temperature_.min()) / (temperature_.max() - temperature_.min())
        loss = temperature_ * (- torch.log(pos_logits / (Ng * np.exp(eta_))))

        return loss


class PairwiseLogisticHardLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticHardLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("margin is {}".format(self._margin))

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.sum(dim=-1)
        # assert pos_logits < Ng
        loss = (- torch.log(pos_logits / (pos_logits + Ng)))
        # loss = (- torch.log(pos_logits / Ng))

        return loss, 0.

class PairwiseLogisticHard_2Loss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticHard_2Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        print("margin is {}".format(self._margin))

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        Ng = neg_logits.sum(dim=-1)
        assert pos_logits < Ng
        loss = (- torch.log(pos_logits / (Ng)))
        # loss = (- torch.log(pos_logits / Ng))

        return loss, 0.

class PairwiseLogisticWeightLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticWeightLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] /  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        # log out
        pos_logits_ = torch.exp(y_pred[:, 0])  #  B
        neg_logits_ = torch.exp(y_pred[:, 1:])  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng_ = neg_logits_.sum(dim=-1)
        loss_ = torch.stack([y_pred[:, 0].detach(), (pos_logits_ / Ng_).detach(), 
                    (- torch.log(pos_logits_ / Ng_)).detach()], dim=0)
        # loss_ = (- torch.log(pos_logits_ / Ng_)).detach()

        return loss, loss_

class PairwiseLogisticV3WeightLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticV3WeightLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] *  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)
        bar_neg = torch.quantile(y_pred[:, 1:], self._negative_weight, dim=1)
        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        # log out
        pos_logits_ = torch.exp(y_pred[:, 0])  #  B
        neg_logits_ = torch.exp(y_pred[:, 1:])  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng_ = neg_logits_.sum(dim=-1)

        loss_ = (- torch.log(pos_logits_ / Ng_)).detach()

        return loss, loss_

class PairwiseLogisticV4WeightLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticV4WeightLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] *  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        # log out
        pos_logits_ = torch.exp(y_pred[:, 0])  #  B
        neg_logits_ = torch.exp(y_pred[:, 1:])  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng_ = neg_logits_.sum(dim=-1)

        loss_ = (- torch.log(pos_logits_ / Ng_)).detach()

        return loss, loss_

class PairwiseLogisticV5WeightLoss(nn.Module):
    def __init__(self, margin=0, temperature=1., negative_weight=None):
        super(PairwiseLogisticV5WeightLoss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight

    def forward(self, y_pred, mask_zeros, temperature_):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] *  temperature_)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_)  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        
        return loss

class PairwiseLogisticInforNCELoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticInforNCELoss, self).__init__()

    def forward(self, y_pred, mask):
        batch_size = y_pred.size(0)
        pos_logits = y_pred.masked_select(~mask).unsqueeze(-1)
        neg_logits = y_pred.masked_select(mask).view(batch_size, -1)
        logits_diff = pos_logits - neg_logits
        loss = - torch.log(torch.sigmoid(logits_diff)).mean()
        return loss


class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(PairwiseMarginLoss, self).__init__()
        self._margin = margin

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        loss = torch.relu(self._margin + neg_logits - pos_logits).mean()
        return loss

class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SigmoidCrossEntropyLoss, self).__init__()
        self._temperature = temperature

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result
        """
        logits = y_pred.flatten() / self._temperature
        labels = y_true.flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        return loss, 0

class SigmoidCrossEntropyWeightLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SigmoidCrossEntropyWeightLoss, self).__init__()
        self._margin = margin 
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true, mask_zeros):
        """
        :param y_true: Labels
        :param y_pred: Predicted result
        """
        y_pred[:, 1:] = torch.where(y_pred[:, 1:] > self._margin, y_pred[:, 1:], mask_zeros)
        # neg_logits = neg_logits[mask
        logits = y_pred.flatten()
        labels = y_true.flatten()

        loss = F.binary_cross_entropy(logits, labels, reduction="sum")
        return loss

class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        probs = F.softmax(y_pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss

