# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE loss.

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='mse_loss',
                ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        self.cls_criterion = nn.MSELoss(reduction=reduction)
    
    def get_name(self):
        return 'MSE_loss'

    def forward(self,
                cls_score,
                reg_label,
                **kwargs):

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            reg_label
        )
        return loss_cls


@LOSSES.register_module()
class CCCLoss(nn.Module):
    """CCC loss for VA regression
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def get_name(self):
        return 'CCC_loss'

    def forward(self,
                cls_score,
                reg_label,
                **kwargs):

        x, y = cls_score, reg_label
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))+1e-10)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)+1e-10)
        loss = 1 - ccc
        return loss * self.loss_weight

