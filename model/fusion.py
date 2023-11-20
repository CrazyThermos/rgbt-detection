import math
import torch
import torch.nn as nn

from model.common import Conv, SPPF
from model.backbone import yolov5_backbone_block
from model.neck import Yolov5Neck
from model.fuseblock import fuse_block_conv1x1

class layer_fusion_1(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024) -> None:
        super().__init__()
        self.gd = gd
        self.gw = gw
        self.rgb_conv_1 = Conv(ch, self.gw_div(last_ch//16), 6, 2, 2)
        self.rgb_block1 = yolov5_backbone_block(last_ch//16, last_ch//8, n=3, gd=gd, gw=gw)
        self.rgb_block2 = yolov5_backbone_block(last_ch//8, last_ch//4, n=6, gd=gd, gw=gw)
        self.rgb_block3 = yolov5_backbone_block(last_ch//4, last_ch//2, n=9, gd=gd, gw=gw)
        self.rgb_block4 = yolov5_backbone_block(last_ch//2, last_ch, n=3, gd=gd, gw=gw)
        # self.rgb_sppf   = SPPF(self.gw_div(last_ch), self.gw_div(last_ch), 5)

        self.t_conv_1 = Conv(ch, self.gw_div(last_ch//16), 6, 2, 2)
        self.t_block1 = yolov5_backbone_block(last_ch//16, last_ch//8, n=3, gd=gd, gw=gw)
        self.t_block2 = yolov5_backbone_block(last_ch//8, last_ch//4, n=6, gd=gd, gw=gw)
        self.t_block3 = yolov5_backbone_block(last_ch//4, last_ch//2, n=9, gd=gd, gw=gw)
        self.t_block4 = yolov5_backbone_block(last_ch//2, last_ch, n=3, gd=gd, gw=gw)
        # self.t_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.fuse_block1 = fuse_block_conv1x1(self.gw_div(last_ch//8))
        self.fuse_block2 = fuse_block_conv1x1(self.gw_div(last_ch//4))
        self.fuse_block3 = fuse_block_conv1x1(self.gw_div(last_ch//2))
        self.fuse_block4 = fuse_block_conv1x1(self.gw_div(last_ch))
    
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
        out1 = rgb_f2 + t_f2
        fuse_f2 = self.fuse_block2(rgb_f2, t_f2)
        
        rgb_f3 = self.rgb_block3(rgb_f2 + fuse_f2)
        t_f3 = self.t_block3(t_f2 + fuse_f2)
        out2 = rgb_f3 + t_f3
        fuse_f3 = self.fuse_block3(rgb_f3, t_f3)
        
        rgb_f4 = self.rgb_block4(rgb_f3 + fuse_f3)
        t_f4 = self.t_block4(t_f3 + fuse_f3)
        out3 = rgb_f4 + t_f4

        return out1, out2, out3

class layer_fusion_2(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0) -> None:
        super().__init__()
    
    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, rgb, t):
        return


def create_model():
    pass


def dump_feature_map(layer, prefix, feature):
    for i in layer:
        plt.imsave("feature_"+prefix+"_"+str(layer)+"_out.png", feature[i][0].transpose(0,1).sum(1).detach().numpy())
    



