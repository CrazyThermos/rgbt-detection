import torch
import torch.nn as nn
from model.mamba.mamba import PatchEmbed, Mamba, MambaConfig

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
        self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1), bias=False)
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
        
        # x = rgb + t
        c = torch.cat((rgb, t), 1)
        c = self.conv1x1(c)
        _, _, h, w = c.size()

        rgb_h = self.avg_pool_x(rgb)
        rgb_w = self.avg_pool_y(rgb).permute(0, 1, 3, 2)

        rgb_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((rgb_h, rgb_w), 2)))

        rgb_cat_conv_split_h, rgb_cat_conv_split_w = rgb_cat_conv_relu.split([h, w], 2)

        rgb_s_h = self.sigmoid_h(self.F_h(rgb_cat_conv_split_h))
        rgb_s_w = self.sigmoid_w(self.F_w(rgb_cat_conv_split_w.permute(0, 1, 3, 2)))

        # _, _, h, w = rgb.size() 
        t_h = self.avg_pool_x(t)
        t_w = self.avg_pool_y(t).permute(0, 1, 3, 2)

        t_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((t_h, t_w), 2)))

        t_cat_conv_split_h, t_cat_conv_split_w = t_cat_conv_relu.split([h, w], 2)


        t_s_h = self.sigmoid_h(self.F_h(t_cat_conv_split_h))
        t_s_w = self.sigmoid_w(self.F_w(t_cat_conv_split_w.permute(0, 1, 3, 2)))
        
        out = c * (rgb_s_h * rgb_s_w + t_s_h * t_s_w)

        return out
    
class fuse_block_CA_v2(nn.Module):
    def __init__(self, ch, reduction=16):
        super(fuse_block_CA_v2, self).__init__()

        # self.h = h
        # self.w = w
        self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1), bias=False)
        self.avg_pool_x1 = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y1 = nn.AdaptiveAvgPool2d((1, None))

        self.avg_pool_x2 = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y2 = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1_1 = nn.Conv2d(in_channels=ch, out_channels=ch//reduction, kernel_size=1, stride=1, bias=False)
        self.conv_1x1_2 = nn.Conv2d(in_channels=ch, out_channels=ch//reduction, kernel_size=1, stride=1, bias=False)

        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(ch//reduction)

        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(ch//reduction)

        self.F_h1 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_w1 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_h2 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_w2 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h1 = nn.Sigmoid()
        self.sigmoid_w1 = nn.Sigmoid()
        self.sigmoid_h2 = nn.Sigmoid()
        self.sigmoid_w2 = nn.Sigmoid()


    def forward(self, rgb, t):
        
        # c = rgb + t
        c = torch.cat((rgb, t), 1)
        c = self.conv1x1(c)
        _, _, h, w = c.size()

        rgb_h = self.avg_pool_x1(rgb)
        rgb_w = self.avg_pool_y1(rgb).permute(0, 1, 3, 2)

        rgb_cat_conv_relu = self.relu1(self.bn1(self.conv_1x1_1(torch.cat((rgb_h, rgb_w), 2))))

        rgb_cat_conv_split_h, rgb_cat_conv_split_w = rgb_cat_conv_relu.split([h, w], 2)

        rgb_s_h = self.sigmoid_h1(self.F_h1(rgb_cat_conv_split_h))
        rgb_s_w = self.sigmoid_w1(self.F_w1(rgb_cat_conv_split_w.permute(0, 1, 3, 2)))

        # _, _, h, w = rgb.size() 
        t_h = self.avg_pool_x2(t)
        t_w = self.avg_pool_y2(t).permute(0, 1, 3, 2)

        t_cat_conv_relu = self.relu2(self.conv_1x1_2(torch.cat((t_h, t_w), 2)))

        t_cat_conv_split_h, t_cat_conv_split_w = t_cat_conv_relu.split([h, w], 2)

        t_s_h = self.sigmoid_h2(self.F_h2(t_cat_conv_split_h))
        t_s_w = self.sigmoid_w2(self.F_w2(t_cat_conv_split_w.permute(0, 1, 3, 2)))
        
        out = c * (rgb_s_h * rgb_s_w + t_s_h * t_s_w)

        return out


class fuse_block_CA_v3(nn.Module):
    def __init__(self, ch, reduction=16):
        super(fuse_block_CA_v3, self).__init__()

        # self.h = h
        # self.w = w
        # self.conv1x1 = nn.Conv2d(ch * 2, ch,(1,1), bias=False)
        self.avg_pool_x1 = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y1 = nn.AdaptiveAvgPool2d((1, None))

        self.avg_pool_x2 = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y2 = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1_1 = nn.Conv2d(in_channels=ch, out_channels=ch//reduction, kernel_size=1, stride=1, bias=False)
        self.conv_1x1_2 = nn.Conv2d(in_channels=ch, out_channels=ch//reduction, kernel_size=1, stride=1, bias=False)

        self.act1 = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(ch//reduction)

        self.act2 = nn.SiLU()
        self.bn2 = nn.BatchNorm2d(ch//reduction)

        self.F_h1 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_w1 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_h2 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)
        self.F_w2 = nn.Conv2d(in_channels=ch//reduction, out_channels=ch, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h1 = nn.Sigmoid()
        self.sigmoid_w1 = nn.Sigmoid()
        self.sigmoid_h2 = nn.Sigmoid()
        self.sigmoid_w2 = nn.Sigmoid()

    def forward(self, rgb, t):
        
        # c = rgb + t
        # c = torch.cat((rgb, t), 1)
        # c = self.conv1x1(c)
        _, _, h, w = t.size()

        rgb_h = self.avg_pool_x1(rgb)
        rgb_w = self.avg_pool_y1(rgb).permute(0, 1, 3, 2)

        rgb_cat_conv_relu = self.act1(self.bn1(self.conv_1x1_1(torch.cat((rgb_h, rgb_w), 2))))

        rgb_cat_conv_split_h, rgb_cat_conv_split_w = rgb_cat_conv_relu.split([h, w], 2)

        rgb_s_h = self.sigmoid_h1(self.F_h1(rgb_cat_conv_split_h))
        rgb_s_w = self.sigmoid_w1(self.F_w1(rgb_cat_conv_split_w.permute(0, 1, 3, 2)))

        # _, _, h, w = rgb.size() 
        t_h = self.avg_pool_x2(t)
        t_w = self.avg_pool_y2(t).permute(0, 1, 3, 2)

        t_cat_conv_relu = self.act2(self.conv_1x1_2(torch.cat((t_h, t_w), 2)))

        t_cat_conv_split_h, t_cat_conv_split_w = t_cat_conv_relu.split([h, w], 2)

        t_s_h = self.sigmoid_h2(self.F_h2(t_cat_conv_split_h))
        t_s_w = self.sigmoid_w2(self.F_w2(t_cat_conv_split_w.permute(0, 1, 3, 2)))
        
        out = rgb * (rgb_s_h * rgb_s_w + t_s_h * t_s_w) + t
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