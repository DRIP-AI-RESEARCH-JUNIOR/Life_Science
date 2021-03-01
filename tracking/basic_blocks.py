import torch
from torch import nn

import numpy as np

def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


def center_crop7(x):
    """
    Center crop layer for stage1 of resnet. (7*7)
    input x can be a Variable or Tensor
    """

    return x[:, :, 2:-2, 2:-2].contiguous()

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)
    
class Bottleneck_CI(nn.Module):
    """
    Bottleneck with center crop layer, utilized in CVPR2019 model
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

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

        out += residual

        if self.last_relu:         # remove relu for the last block
            out = self.relu(out)

        out = center_crop(out)     # in-residual crop

        return out
    
    class Bottleneck_BIG_CI(nn.Module):
    """
    Bottleneck with center crop layer, double channels in 3*3 conv layer in shortcut branch
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_BIG_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

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

        out += residual

        if self.last_relu:  # feature out no relu
            out = self.relu(out)

        out = center_crop(out)  # in-layer crop

        return out

class ResNet(nn.Module):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # stage2
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])

        # stage3
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop7(x)
        x = self.maxpool(x)   # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)    # stride = 8

        return x

class BasicConv2d_1x1(nn.Module):
    """
    1*1 branch of inception
    """

    def __init__(self, in_channels, out_channels, last_relu=True, **kwargs):
        super(BasicConv2d_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.last_relu:
            return F.relu(x, inplace=True)
        else:
            return x


class BasicConv2d_3x3(nn.Module):
    """
    3*3 branch of inception
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=True):
        super(BasicConv2d_3x3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.last_relu:
            out = self.relu(out)

        return out


class InceptionM(nn.Module):
    """
    Inception module with 1*1 and 3*3 branch
    """

    def __init__(self, in_channels, planes, last_relu=True):
        super(InceptionM, self).__init__()
        self.branch3x3 = BasicConv2d_3x3(in_channels, planes, last_relu)
        self.branch1x1 = BasicConv2d_1x1(in_channels, planes, last_relu, kernel_size=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch1x1 = self.branch1x1(x)

        outputs = [branch3x3, branch1x1]
        return center_crop(torch.cat(outputs, 1))   # in-layer crop


class Inception(nn.Module):
    """
    Inception with 22 layer utilized in CVPR2019 paper.
    Usage: Inception(InceptionM, [3, 4], [True, False])
    """

    def __init__(self, block, layers):
        self.inplanes = 64
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, 64, layers[0], pool=False)     # in=64, out=320
        self.layer2 = self._make_layer(block, 320, 128, layers[1], pool=True, last_relu=False)   # in=320, out=640

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inchannels, planes, blocks, pool=True, last_relu=True):

        layers = []
        for i in range(0, blocks):
            if i == 0:
                self.inchannels = inchannels
            else:
                self.inchannels = planes * 5

            if i == 1 and pool:
                layers.append(self.maxpool)

            if i == blocks - 1 and not last_relu:
                layers.append(block(self.inchannels, planes, last_relu))
            else:
                layers.append(block(self.inchannels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x
