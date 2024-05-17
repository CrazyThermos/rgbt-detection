import math
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from model.common import C3, Conv, SPPF
from model.mamba import Mamba, MambaConfig, PatchMerge, RMSNorm
from model.vmamba import VSSBlock
from model.light_mamba import Light_VSSBlock
# YOLOv5 v6.0 backbone
'''
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]
'''
class yolov5_backbone(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0) -> None:
        super().__init__()
        self.gd = gd
        self.gw = gw
        self.conv_1 = Conv(ch, self.gw_div(64), 4, 2, 2)
        self.conv_2 = Conv(self.gw_div(64), self.gw_div(128), 3, 2)
        self.c3_1   =   C3(self.gw_div(128), self.gw_div(128), self.gd_muti(3))
        self.conv_3 = Conv(self.gw_div(128), self.gw_div(256), 3, 2)
        self.c3_2   =   C3(self.gw_div(256), self.gw_div(256), self.gd_muti(6))
        self.conv_4 = Conv(self.gw_div(256), self.gw_div(512), 3, 2)
        self.c3_3   =   C3(self.gw_div(512), self.gw_div(512), self.gd_muti(9))
        self.conv_5 = Conv(self.gw_div(512), self.gw_div(1024), 3, 2)
        self.c3_4   =   C3(self.gw_div(1024), self.gw_div(1024), self.gd_muti(3))
        self.sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)
        self.net    = nn.Sequential(self.conv_1, self.conv_2, self.c3_1, 
                                    self.conv_3, self.c3_2, self.conv_4,
                                    self.c3_3, self.conv_5, self.c3_4, 
                                    self.sppf)
        for block in self.net:
            if type(block) == Conv:
                nn.init.kaiming_normal_(block.conv)
            elif type(block) == C3:
                nn.init.kaiming_normal_(block.cv1.conv)
                nn.init.kaiming_normal_(block.cv2.conv)
                nn.init.kaiming_normal_(block.cv3.conv)
                for layer in block.m:
                    nn.init.kaiming_normal_(layer.cv1.conv)
                    nn.init.kaiming_normal_(layer.cv2.conv)
            elif type(block) == SPPF:
                nn.init.kaiming_normal_(block.cv1.conv)
                nn.init.kaiming_normal_(block.cv2.conv)


    def gd_muti(self, n):
        gd = self.gd
        return max(round(n * gd), 1) if n > 1 else n

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, x):
        o = self.net(x)
        return o

class yolov5_backbone_block(nn.Module):
    def __init__(self, ch_in, ch_out, n=1, gd=1.0, gw=1.0) -> None:
        super().__init__()
        self.gd = gd
        self.gw = gw

        self.conv = Conv(self.gw_div(ch_in), self.gw_div(ch_out), 3, 2)
        self.c3  =   C3(self.gw_div(ch_out), self.gw_div(ch_out), self.gd_muti(n))

    def gd_muti(self, n):
        gd = self.gd
        return max(round(n * gd), 1) if n > 1 else n

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, x):
        o = self.c3(self.conv(x))
        return o

class mamba_backbone_block(nn.Module):
    def __init__(self, 
        dim,  # # 96
        depth,  # 2
        d_state=16,
        mamba_drop=0.,
        drop_path=0.,   # 每一个模块都有一个drop
        norm_layer=nn.LayerNorm, 
        downsample=None,  # PatchMergin2D
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Light_VSSBlock(
                hidden_dim=dim,   # 96
                drop_path=drop_path[i],  # 0.2
                norm_layer=norm_layer,  # nn.LN
                d_state=d_state,  # 16
            ) for i in range(depth)])
        # self.layers = nn.ModuleList([
        #     VSSBlock(
        #         hidden_dim=dim,   # 96
        #         drop_path=drop_path[i],  # 0.2
        #         norm_layer=norm_layer,  # nn.LN
        #         d_state=d_state,  # 16
        #     ) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.downsample is not None:
            x_ = self.downsample(x.permute(0, 2, 3, 1))            
        else:
            x_ = None
        return (x_, x)
