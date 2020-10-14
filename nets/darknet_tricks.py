import torch
import torch.nn as nn
import math
from numpy import *
import torch.nn.functional as F
from collections import OrderedDict


#   卷积块
#   CONV+BATCHNORM+LeakReLU
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.conv(x)


# 内部堆叠的残差块
class ConvResidual(nn.Module):
    def __init__(self, channels, hidden_channels):
        super(ConvResidual, self).__init__()
        hidden_channels = channels // 2

        self.block = nn.Sequential(
            Conv(channels, hidden_channels, 1, 1, 0),
            Conv(hidden_channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)  # 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64)  # 一个残差块 = 2层卷积
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(  # = 4层卷积
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26 = nn.Sequential(  # = 16层卷积
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
        )
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43 = nn.Sequential(  # = 16层卷积
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
        )
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52 = nn.Sequential(  # = 8层卷积
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
        )

    def forward(self, entry):
        conv1 = self.conv1(entry)
        conv2 = self.conv2(conv1)
        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26(conv10)
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43(conv27)
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52(conv44)
        return conv45_52, conv28_43, conv11_26  # YOLOv3用，所以输出了3次特征


def darknet53(pretrained, **kwargs):
    model = Darknet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
