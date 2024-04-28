import math
import torch
import torch.nn as nn

from model.common import *

# YOLOv5 v6.0 head
# head:
#   [[-1, 1, Conv, [512, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
#    [-1, 3, C3, [512, False]],  # 13

#    [-1, 1, Conv, [256, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
#    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

#    [-1, 1, Conv, [256, 3, 2]],
#    [[-1, 14], 1, Concat, [1]],  # cat head P4
#    [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

#    [-1, 1, Conv, [512, 3, 2]],
#    [[-1, 10], 1, Concat, [1]],  # cat head P5
#    [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

#    [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
#   ]

class Yolov5Neck(nn.Module):
    def __init__(self, ch, n=3, gd=1.0, gw=1.0, last_ch=1024):
        super().__init__()
        self.gd = gd
        self.gw = gw
        self.conv_1 = Conv(self.gw_div(ch), self.gw_div(last_ch//2), 1, 1)
        self.upsample_1 = nn.Upsample(None, 2)
        self.c3_1 = C3(self.gw_div(last_ch), self.gw_div(last_ch//2), self.gd_muti(n)) 

        self.conv_2 = Conv(self.gw_div(last_ch//2), self.gw_div(last_ch//4), 1, 1)
        self.upsample_2 = nn.Upsample(None, 2)
        self.c3_2 = C3(self.gw_div(last_ch//2), self.gw_div(last_ch//4), self.gd_muti(n)) 

        self.conv_3 = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 3, 2)
        self.c3_3 = C3(self.gw_div(last_ch//2), self.gw_div(last_ch//2), self.gd_muti(n)) 

        self.conv_4 = Conv(self.gw_div(last_ch//2), self.gw_div(last_ch//2), 3, 2)
        self.c3_4 = C3(self.gw_div(last_ch), self.gw_div(last_ch), self.gd_muti(n)) 

    def gd_muti(self, n):
        gd = self.gd
        return max(round(n * gd), 1) if n > 1 else n

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, x1, x2, x3):
        conv_out1 = self.conv_1(x3)
        conv_out2 = self.conv_2(self.c3_1(torch.concat((x2,self.upsample_1(conv_out1)), 1)))
        out1 = self.c3_2(torch.concat((x1, self.upsample_2(conv_out2)), 1))
        out2 = self.c3_3(torch.concat((conv_out2, self.conv_3(out1)), 1))
        out3 = self.c3_4(torch.concat((conv_out1, self.conv_4(out2)), 1))
        return out1, out2, out3


class RTDETRNeck(nn.Module):
    def __init__(self, ch, n=3, gd=1.0, gw=1.0, last_ch=1024):
        super().__init__()
        self.gd = gd
        self.gw = gw

        self.conv_0 = Conv(self.gw_div(ch), self.gw_div(last_ch//4), 1, 1, None, 1, 1, False) #input3
        self.aifi = AIFI(self.gw_div(last_ch//4), cm=1024)
        self.conv_0_act = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 1, 1)

        self.upsample_1 = nn.Upsample(None, 2)
        self.conv_1 = Conv(self.gw_div(last_ch//2), self.gw_div(last_ch//4), 1, 1, None, 1, 1, False) #input2
        self.c3_1 = RepC3(self.gw_div(last_ch//2), self.gw_div(last_ch//4), self.gd_muti(n)) 
        self.conv_1_act = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 1, 1)

        self.upsample_2 = nn.Upsample(None, 2)
        self.conv_2 = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 1, 1, None, 1, 1, False) #input1
        self.c3_2 = RepC3(self.gw_div(last_ch//2), self.gw_div(last_ch//4), self.gd_muti(n)) #output1

        self.conv_3 = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 3, 2)
        self.c3_3 = RepC3(self.gw_div(last_ch//2), self.gw_div(last_ch//4), self.gd_muti(n)) #output2

        self.conv_4 = Conv(self.gw_div(last_ch//4), self.gw_div(last_ch//4), 3, 2)
        self.c3_4 = RepC3(self.gw_div(last_ch//2), self.gw_div(last_ch//4), self.gd_muti(n)) #output3

    def gd_muti(self, n):
        gd = self.gd
        return max(round(n * gd), 1) if n > 1 else n

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward(self, x1, x2, x3):
        conv_out1 = self.conv_0_act(self.aifi(self.conv_0(x3))) # 12
        conv_out2 = self.conv_1_act(self.c3_1(torch.concat((self.conv_1(x2), self.upsample_1(conv_out1)), 1))) # 17
        out1 = self.c3_2(torch.concat((self.conv_2(x1), self.upsample_2(conv_out2)), 1))
        out2 = self.c3_3(torch.concat((conv_out2, self.conv_3(out1)), 1))
        out3 = self.c3_4(torch.concat((conv_out1, self.conv_4(out2)), 1))
        return out1, out2, out3