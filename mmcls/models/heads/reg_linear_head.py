# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_loss
from .base_head import BaseHead
from ..utils import is_tracing

@HEADS.register_module()
class LinearRegHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 dims=(2048, 1),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(loss, dict):
            loss = [loss]
        self.compute_losses = [build_loss(i) for i in loss]
        self.fcs = self.build_fcs(dims)
    
    def build_fcs(self, dims):
        fcs = []
        for i in range(1, len(dims)-1):
            fcs += [
                nn.Linear(dims[i - 1], dims[i]),
                nn.BatchNorm1d(dims[i]),
                nn.ReLU()
            ]
        fcs.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*fcs)  # skip the last BN and ReLU

    def loss(self, cls_score, reg_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        for loss_func in self.compute_losses:
            losses[loss_func.get_name()] = loss_func(cls_score, reg_label, avg_factor=num_samples, **kwargs)
        return losses
    
    def linear_transpose(self, cls_score):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        cls_score = self.fcs(cls_score)
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        return cls_score

    def forward_train(self, cls_score, gt_label, **kwargs):
        cls_score = self.linear_transpose(cls_score)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, cls_score, post_process=True):
        """Inference without augmentation.

        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        pred = self.linear_transpose(cls_score)

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

