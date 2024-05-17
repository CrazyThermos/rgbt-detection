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
    def __init__(self, output = 1):
        super(FusionDecoder, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        # output=1
        # self.vis_conv = ConvLeakyRelu2d(output, vis_ch[0])
        # self.vis_grm1 = GRM(vis_ch[0], vis_ch[1])
        # self.vis_grm2 = GRM(vis_ch[1], vis_ch[2])
        # self.vis_se = SEBlock(vis_ch[2])

        # self.inf_conv = ConvLeakyRelu2d(output, inf_ch[0])
        # self.inf_grm1 = GRM(inf_ch[0], inf_ch[1])
        # self.inf_grm2 = GRM(inf_ch[1], inf_ch[2])
        # self.inf_se = SEBlock(inf_ch[2])

        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
        
    def forward(self, feature_vis, feature_ir):
        # split data into RGB and INF
        # x_vis_origin = image_vis
        # x_inf_origin = image_ir
        # encode
        # x_vis_p=self.vis_conv(x_vis_origin)
        # x_vis_p1=self.vis_grm1(x_vis_p)
        # x_vis_p2=self.vis_grm2(x_vis_p1)
        # x_vis_p2=self.vis_se(x_vis_p2)

        # x_inf_p=self.inf_conv(x_inf_origin)
        # x_inf_p1=self.inf_grm1(x_inf_p)
        # x_inf_p2=self.inf_grm2(x_inf_p1)
        # x_inf_p2=self.inf_se(x_inf_p2)

        # decode
        x=self.decode4(torch.cat((feature_vis, feature_ir),dim=1))
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x
