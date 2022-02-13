# Copyright (c) OpenMMLab. All rights reserved.
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
                ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight


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
