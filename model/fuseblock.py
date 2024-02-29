import torch
import torch.nn as nn

class fuse_block_conv1x1(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1))
    def forward(self, rgb_feature, t_feature):
        c = torch.cat((rgb_feature, t_feature), 1)
        out = self.conv1x1(c)
        return out

class fuse_block_discriminative(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()

    def forward(self, rgb, t):
        pass

class fuse_block_complementary(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()

    def forward(self, rgb, t):
        pass

class fuse_block_AFF(nn.Module):
    def __init__(self, ch, r=4) -> None:
        super().__init__()
        in_ch = ch//r

        self.local_att = nn.Sequential(
            nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, rgb, t):
        f_a = rgb + t
        f_local = self.local_att(f_a)
        f_global = self.global_att(f_a)
        f_lg = f_local + f_global
        weight = self.sigmoid(f_lg)
        out = 2*(rgb*weight + t*(1-weight))
        return out