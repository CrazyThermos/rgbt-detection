import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5
    
class FusionDecoder(nn.Module):
    def __init__(self, ch=1024):
        super(FusionDecoder, self).__init__()
        vis_ch = [ch//4,ch//2,ch]

        self.upsample4 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1], vis_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], 3)
        
    def forward(self, x):
        # decode
        x=self.upsample4(x)
        x=self.decode4(x)
        x=self.upsample3(x)
        x=self.decode3(x)
        x=self.upsample2(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x
