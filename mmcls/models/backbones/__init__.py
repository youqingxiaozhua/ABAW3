# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .conformer import Conformer
from .convnext import ConvNeXt
from .deit import DistilledVisionTransformer
from .efficientnet import EfficientNet
from .hrnet import HRNet
from .lenet import LeNet5
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .regnet import RegNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin_transformer import SwinTransformer
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tnt import TNT
from .twins import PCPVT, SVT
from .vgg import VGG
from .vision_transformer import VisionTransformer
from .irse import IRSE
from .iresnet import IResNet
from .swin_faceX import SwinTransformerFaceX
from .vgg_face import VGGFace
from .repvgg_facex import RepVGGFaceX
from .attention_facex import ResidualAttentionNet
from .resnest_facex import ResNeStFaceX
from .hrnet_facex import HRNetFaceX

__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone', 'T2T_ViT', 'Res2Net', 'RepVGG',
    'Conformer', 'MlpMixer', 'DistilledVisionTransformer', 'PCPVT', 'SVT',
    'EfficientNet', 'ConvNeXt', 'HRNet', 'IRSE', 'IResNet', 'SwinTransformerFaceX',
    'VGGFace', 'ResidualAttentionNet', 'ResNeStFaceX', 'HRNetFaceX'
]
