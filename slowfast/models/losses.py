#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

## Added
import pandas as pd
import numpy as np

dir_action_count = '/mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/annotation/df_action.xlsx'

class BCELoss(nn.Module):
    '''
    Function: BCELoss
    Params:
        predictions: input->(batch_size, 1004)
        targets: target->(batch_size, 1004)
    Return:
        bceloss
    '''

    def __init__(self,logits=True, reduction="mean"):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)

        return BCE_loss

class FocalLoss(nn.Module):
    '''
    Function: FocalLoss
    Params:
        alpha: scale factor, default = 1
        gamma: exponential factor, default = 0
    Return:
        focalloss
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/facebookresearch/Detectron
    '''

    def __init__(self, logits=True, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = 1 
        self.gamma = 0 
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


class LDAM(nn.Module):
    '''
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    '''

    def __init__(self, logits=True, reduction='mean', max_m=0.5, s=30, step_epoch=80):
        super(LDAM, self).__init__()

        data = pd.read_excel(dir_action_count)
        self.num_class_list = list(map(float, data["count"].tolist()))  
        self.reduction = reduction
        self.logits = logits

        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        self.s = s
        self.step_epoch = step_epoch
        self.weight = None

    def reset_epoch(self, epoch):
        idx = epoch // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).cuda()

    def forward(self, inputs, targets):
        targets=targets.to(torch.float32)
        batch_m = torch.matmul(self.m_list[None, :], targets.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        inputs_m = inputs - batch_m

        output = torch.where(targets.type(torch.uint8), inputs_m, inputs)
        if self.logits:
            loss = F.binary_cross_entropy_with_logits(self.s * output, targets, reduction=self.reduction,
                                                    weight=self.weight)
        else:
            loss = F.binary_cross_entropy(self.s * output, targets, reduction=self.reduction, weight=self.weight)
        return loss


class EQL(nn.Module):
    '''
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/tztztztztz/eql.detectron2
    '''

    def __init__(self, logits=True, reduction='mean', max_tail_num=100, gamma=1.76 * 1e-3):
        super(EQL, self).__init__()
        data = pd.read_excel(dir_action_count)
        num_class_list = list(map(float, data["count"].tolist())) 
        self.reduction = reduction
        self.logits = logits

        max_tail_num = max_tail_num
        self.gamma = gamma

        self.tail_flag = [False] * len(num_class_list)
        for i in range(len(self.tail_flag)):
            if num_class_list[i] <= max_tail_num:
                self.tail_flag[i] = True

    def threshold_func(self):
        weight = self.inputs.new_zeros(self.n_c)
        weight[self.tail_flag] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def beta_func(self):
        rand = torch.rand((self.n_i, self.n_c)).cuda()
        rand[rand < 1 - self.gamma] = 0
        rand[rand >= 1 - self.gamma] = 1
        return rand

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.n_i, self.n_c = self.inputs.size()

        eql_w = 1 - self.beta_func() * self.threshold_func() * (1 - targets)
        if self.logits:
            loss = F.binary_cross_entropy_with_logits(self.inputs, targets, reduction=self.reduction, weight=eql_w)
        else:
            loss = F.binary_cross_entropy(self.inputs, targets, reduction=self.reduction, weight=eql_w)
        return loss

class EQLv2(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 test_with_obj=True):
        super().__init__()
        data = pd.read_excel(dir_action_count)
        num_class_list = list(map(float, data["count"].tolist())) 
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = len(num_class_list)
        self.group = True

        # cfg for eqlv2
  
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes).cuda())
        self.register_buffer('neg_grad', torch.zeros(self.num_classes).cuda())
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', (torch.ones(self.num_classes) * 100).cuda())

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self,
                cls_score,
                label,
                weight=None):
        self.n_i, self.n_c = cls_score.size()

        target = label
        self.pred_class_logits = cls_score

        pos_w, neg_w = self.get_weight()
        #print(pos_w.device, target.device)
        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)
        #print(self.pos_grad.device, pos_grad.device)
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self):
        neg_w = self.map_func(self.pos_neg)
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w.cuda(), neg_w.cuda()

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "bce_loss": BCELoss,
    "focal_loss": FocalLoss,
    "LDAM": LDAM,
    "EQL": EQL,
    "EQLv2": EQLv2
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

# if __name__ == "__main__":
#    preds = model(inputs)
#    loss_fun = get_loss_func(cfg.MODEL.LOSS_FUNC)()
#    #Uncomment the following line if you want to utilize LDAM-DRW.
#    # loss_fun.reset_epoch(cur_epoch)
#    loss = loss_fun(preds, labels)
