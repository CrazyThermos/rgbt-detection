import math
import torch
import torch.nn as nn

from model.common import Conv, SPPF
from model.backbone import yolov5_backbone_block
from model.neck import Yolov5Neck
from model.head import Yolov5DetectHead
from model.fuseblock import fuse_block_conv1x1

class base_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = None
        self.names = None # class names
        self.anchors = []


class rgbt_yolov5(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024, nc=2, training=True) -> None:
        super().__init__()
        self.gd = gd
        self.gw = gw
        self.nc = nc
        self.rgb_conv_1 = Conv(ch, self.gw_div(last_ch//16), 6, 2, 2)
        self.rgb_block1 = yolov5_backbone_block(last_ch//16, last_ch//8, n=3, gd=gd, gw=gw)
        self.rgb_block2 = yolov5_backbone_block(last_ch//8, last_ch//4, n=6, gd=gd, gw=gw)
        self.rgb_block3 = yolov5_backbone_block(last_ch//4, last_ch//2, n=9, gd=gd, gw=gw)
        self.rgb_block4 = yolov5_backbone_block(last_ch//2, last_ch, n=3, gd=gd, gw=gw)
        self.rgb_sppf   = SPPF(self.gw_div(last_ch), self.gw_div(last_ch), 5)

        self.t_conv_1 = Conv(ch, self.gw_div(last_ch//16), 6, 2, 2)
        self.t_block1 = yolov5_backbone_block(last_ch//16, last_ch//8, n=3, gd=gd, gw=gw)
        self.t_block2 = yolov5_backbone_block(last_ch//8, last_ch//4, n=6, gd=gd, gw=gw)
        self.t_block3 = yolov5_backbone_block(last_ch//4, last_ch//2, n=9, gd=gd, gw=gw)
        self.t_block4 = yolov5_backbone_block(last_ch//2, last_ch, n=3, gd=gd, gw=gw)
        self.t_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.fuse_block1 = fuse_block_conv1x1(self.gw_div(last_ch//8))
        self.fuse_block2 = fuse_block_conv1x1(self.gw_div(last_ch//4))
        self.fuse_block3 = fuse_block_conv1x1(self.gw_div(last_ch//2))
        self.fuse_block4 = fuse_block_conv1x1(self.gw_div(last_ch))
    
        self.neck_block = Yolov5Neck(last_ch, n=3, gd=self.gd, gw=self.gw, last_ch=last_ch)
        self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        self.detect_block = Yolov5DetectHead(nc, self.anchors, ch=[int(last_ch/4*self.gw), int(last_ch/2*self.gw), int(last_ch*self.gw)], training=training)

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, rgb, t):
        rgb_f1 = self.rgb_block1(self.rgb_conv_1(rgb))
        t_f1 = self.t_block1(self.t_conv_1(t))
        fuse_f1 = self.fuse_block1(rgb_f1, t_f1)
        
        rgb_f2 = self.rgb_block2(rgb_f1 + fuse_f1)
        t_f2 = self.t_block2(t_f1 + fuse_f1)
        fuse1 = rgb_f2 + t_f2
        fuse_f2 = self.fuse_block2(rgb_f2, t_f2)
        
        rgb_f3 = self.rgb_block3(rgb_f2 + fuse_f2)
        t_f3 = self.t_block3(t_f2 + fuse_f2)
        fuse2 = rgb_f3 + t_f3
        fuse_f3 = self.fuse_block3(rgb_f3, t_f3)
        
        rgb_f4 = self.rgb_block4(rgb_f3 + fuse_f3)
        t_f4 = self.t_block4(t_f3 + fuse_f3)
        rgb_f4 = self.rgb_sppf(rgb_f4)
        t_f4 = self.t_sppf(t_f4)
        fuse3 = rgb_f4 + t_f4
        
        neckout1, neckout2, neckout3 = self.neck_block(fuse1, fuse2, fuse3)
        res = self.detect_block([neckout1, neckout2, neckout3])
        return res



class rgbt_yolov5_2(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0) -> None:
        super().__init__()
    
    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, rgb, t):
        return


RGBTModel = rgbt_yolov5

    



