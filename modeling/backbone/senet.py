from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import math
from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
from torch.utils import model_zoo


from ModelCore.utils.registry import Registry

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1):

        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )


    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x2, x3, x4, x5]



    def forward(self, x):
        outputs = self.features(x)
        return outputs



SENet50 = dict(
    block="SEResNetBottleneck",
    stage_block_num=[3, 4, 6, 3],
    groups=1,
    reduction=16,
    inplanes=64,
    input_3x3=False,
    downsample_kernel_size=1,
    downsample_padding=0,
)

SENet101 = dict(
    block="SEResNetBottleneck",
    stage_block_num=[3, 4, 23, 3],
    groups=1,
    reduction=16,
    inplanes=64,
    input_3x3=False,
    downsample_kernel_size=1,
    downsample_padding=0,
)

SENeXt50 = dict(
    block="SEResNeXtBottleneck",
    stage_block_num=[3, 4, 6, 3],
    groups=32,
    reduction=16,
    inplanes=64,
    input_3x3=False,
    downsample_kernel_size=1,
    downsample_padding=0,
)

SENeXt101 = dict(
    block="SENeXtBottleneck",
    stage_block_num=[3, 4, 23, 3],
    groups=32,
    reduction=16,
    inplanes=64,
    input_3x3=False,
    downsample_kernel_size=1,
    downsample_padding=0,
)

_MODEL_SPECS = Registry({
    "SE-R-50-FPN": SENet50,
    "SE-R-101-FPN": SENet101,
    "SE-X-50-FPN": SENeXt50,
    "SE-X-101-FPN": SENeXt101,
    "SE-R-50-PAN": SENet50,
    "SE-R-101-PAN": SENet101,
    "SE-X-50-PAN": SENeXt50,
    "SE-X-101-PAN": SENeXt101,
})

_TRANSFORMATION_MODULES = Registry({
    "SEBottleneck": SEBottleneck,
    "SEResNetBottleneck": SEResNetBottleneck,
    "SEResNeXtBottleneck": SEResNeXtBottleneck,
})


def load_weight(model, weight_path):
    pretrained_dict = torch.load(weight_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Pre-Trained SENet weight loaded!")
    return model

def build_senet(cfg):
    model_spec = _MODEL_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
    block = _TRANSFORMATION_MODULES[model_spec['block']]
    block_nums = model_spec["stage_block_num"]
    groups = model_spec["groups"]
    reduction = model_spec["reduction"]
    inplanes = model_spec["inplanes"]
    input_3x3 = model_spec["input_3x3"]
    downsample_kernel_size = model_spec["downsample_kernel_size"]
    downsample_padding = model_spec["downsample_padding"]

    model = SENet(block, layers=block_nums, groups=groups, reduction=reduction,
                  inplanes=inplanes, input_3x3=input_3x3,
                  downsample_kernel_size=downsample_kernel_size,
                  downsample_padding=downsample_padding)

    if os.path.isfile(cfg.MODEL.SENET.PRETRAINED_WEIGHT):
        model = load_weight(model, cfg.MODEL.SENET.PRETRAINED_WEIGHT)

    return model






if __name__ == '__main__':
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    pretrained_dict = torch.load('/home/../data/tct/TorchModelZoo/pretrained_models/senet/se_resnext50_32x4d.pth')
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model.state_dict()}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

