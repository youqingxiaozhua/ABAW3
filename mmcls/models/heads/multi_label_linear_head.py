# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead


@HEADS.register_module()
class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelLinearClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred


@HEADS.register_module()
class AUClsHead(MultiLabelClsHead):
    """Linear classification head for action unit task.
        every AU has a separate branch: fc, bn, relu, fc -> 1
        there are num_classes branches
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super().__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.fc1s = nn.Sequential(*[nn.Linear(in_channels, hidden_channels) for _ in range(num_classes)])
        self.bns = nn.Sequential(*[nn.BatchNorm1d(hidden_channels) for _ in range(num_classes)])
        self.relu = nn.ReLU()
        self.fc2s = nn.Sequential(*[nn.Linear(hidden_channels, 1, bias=False) for _ in range(num_classes)])

    def extract_feat(self, x: torch.Tensor):
        au_out = []
        for i in range(self.num_classes):
            au = self.fc1s[i](x)
            au = self.relu(self.bns[i](au))
            au = self.fc2s[i](au)
            au_out.append(au.squeeze(1))
        return torch.stack(au_out, dim=1)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        cls_score = self.extract_feat(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.extract_feat(x)

        if sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

