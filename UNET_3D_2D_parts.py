""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=None, padding=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if not (kernel_size and padding):
            kernel_size = 3
            padding = 1
        self.double_conv = nn.Sequential(OrderedDict([
            ('d2conv1', nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)),
            ('d2batch', nn.BatchNorm2d(mid_channels)),
            ('d2relu1', nn.ReLU(inplace=True)),
            ('d2conv2', nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)),
            ('d2batch2', nn.BatchNorm2d(out_channels)),
            ('d2relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(OrderedDict([
            ('d3conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=(9, 3, 3), stride=1, padding=(4, 1, 1))),
            ('d3batch1', nn.BatchNorm3d(mid_channels)),
            ('d3relu1', nn.ReLU(inplace=True)),
            ('d3conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=(9, 3, 3), stride=1, padding=(4, 1, 1))),
            ('d3batch2', nn.BatchNorm3d(out_channels)),
            ('d3relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool3d((4, 2, 2)),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    先将输入的通道数减半，长宽翻倍，然后与上一层的矩阵合并
    """

    def __init__(self, in_channels, out_channels, frame, bilinear=True):
        super().__init__()

        self.cross = nn.AvgPool3d((frame, 1, 1))

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.cross(x2)
        x2 = x2.squeeze(dim=2)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super(UpSampler, self).__init__()

        if bilinear:
            self.upsam = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                DoubleConv2D(in_channels // 2, out_channels, in_channels // 2)
            )
        else:
            self.upsam = nn.Sequential(
                nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2),
                DoubleConv2D(in_channels // 2, out_channels)
            )

    def forward(self, x):
        return self.upsam(x)