import torch
import torch.nn as nn
from model.mamba import PatchEmbed, Mamba, MambaConfig

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

        # self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1))
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
        # c = torch.cat((rgb, t), 1)
        # f_a = self.conv1x1(c)

        f_local = self.local_att(f_a)
        f_global = self.global_att(f_a)
        f_lg = f_local + f_global
        weight = self.sigmoid(f_lg)
        out = 2*(rgb*weight + t*(1-weight))
        return out

class fuse_block_CA(nn.Module):
    def __init__(self, ch, reduction=16):
        super(fuse_block_CA, self).__init__()

        # self.h = h
        # self.w = w
        self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1))
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=ch, out_channels=ch//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(ch//reduction)

        self.F_h = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, rgb, t):
        
        x = rgb + t
        # c = torch.cat((rgb, t), 1)
        # x = self.conv1x1(c)
        _, _, h, w = x.size() 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out
    
class fuse_block_Mamba(nn.Module):
    def __init__(self, ch, img_size, patch_size, d_model=768, n_layer=8, d_state=16):
        super(fuse_block_Mamba, self).__init__()
        self.patchembd = PatchEmbed(img_size=img_size, patch_size=patch_size ,stride=patch_size, in_chans=ch)

        config = MambaConfig(d_model=d_model, n_layers=n_layer, d_state=d_state)
        self.mamba = Mamba(config)

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        assert H == W, f"Input feature size H:{H}!=W:{W}!"
        rgb_token = self.patchembd(rgb)
        t_token = self.patchembd(t)
        fuse_token = torch.cat((rgb_token, t_token), 1)
        fuse_out = self.mamba(fuse_token)
        fuse_out = fuse_out.unsqueeze(3)
        fuse_out = fuse_out.transpose(0, 3)
        fuse_out = fuse_out.reshape(B*2, C, H, W)
        return (fuse_out[:B], fuse_out[B:])