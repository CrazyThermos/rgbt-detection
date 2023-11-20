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