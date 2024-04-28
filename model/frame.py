import math
import torch
import torch.nn as nn

from model.common import Conv, SPPF
from model.backbone import yolov5_backbone_block, mamba_backbone_block
from model.neck import Yolov5Neck, RTDETRNeck
from model.head import Yolov5DetectHead
from model.fuseblock import fuse_block_conv1x1,fuse_block_AFF,fuse_block_CA
from model.mamba import PatchEmbed, PatchMerge
from model.vmamba import PatchEmbed2D
from model.unireplknet import UniRepLKNetBlock, LayerNorm, partial
from model.replknet import enable_sync_bn, conv_bn_relu, fuse_bn, get_conv2d, get_bn, checkpoint, RepLKNetStage
class base_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = None
        self.names = None # class names
        self.anchors = []


class rgbt_yolov5(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024, nc=2, training=False) -> None:
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


class rgbt_yolov5_AFF(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024, nc=2, training=False) -> None:
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

        self.fuse_block1 = fuse_block_AFF(ch=self.gw_div(last_ch//8))
        self.fuse_block2 = fuse_block_AFF(ch=self.gw_div(last_ch//4))
        self.fuse_block3 = fuse_block_AFF(ch=self.gw_div(last_ch//2))
        self.fuse_block4 = fuse_block_AFF(ch=self.gw_div(last_ch))
    
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


class rgbt_yolov5_CA(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024, nc=2, training=False) -> None:
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

        self.fuse_block1 = fuse_block_CA(ch=self.gw_div(last_ch//8))
        self.fuse_block2 = fuse_block_CA(ch=self.gw_div(last_ch//4))
        self.fuse_block3 = fuse_block_CA(ch=self.gw_div(last_ch//2))
        self.fuse_block4 = fuse_block_CA(ch=self.gw_div(last_ch))
    
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


# class rgbt_Mamba(nn.Module):
#     def __init__(self, ch, patch_size=16, 
#                  depths=[2, 2, 6, 2], dims=[96, 192, 384, 768], 
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, patch_norm=True, norm_layer=nn.LayerNorm,
#                  gd=1.0, gw=1.0, last_ch=1024, nc=2, training=False) -> None:
#         super().__init__()
#         self.gd = gd
#         self.gw = gw
#         self.nc = nc

#         self.patchembd = PatchEmbed2D(patch_size=patch_size, in_chans=ch, embed_dim=dims[0],
#             norm_layer=norm_layer if patch_norm else None)
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
#         self.layers=nn.ModuleList()
#         for i in range(len(depths)):
#             layer = mamba_backbone_block(dim=dims[i],
#                                          depth=depths[i],
#                                          mamba_drop = attn_drop_rate,
#                                          drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
#                                          norm_layer=norm_layer, # nn.LN
#                                          downsample=PatchMerging2D if (i < len(depths) - 1) else None)
#             self.layers.append(layer)
#         self.neck_block = Yolov5Neck(last_ch, n=3, gd=self.gd, gw=self.gw, last_ch=last_ch)
#         self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
#         self.detect_block = Yolov5DetectHead(nc, self.anchors, 
#                                              ch=[int(last_ch/4*self.gw), 
#                                                  int(last_ch/2*self.gw), 
#                                                  int(last_ch*self.gw)], 
#                                                  training=training)

#     def gw_div(self, x):
#         divisor = 8 
#         x *= self.gw
#         return int(math.ceil(x / divisor) * divisor)
    
#     def forward(self, rgb, t):
#         fuse_list = []
#         rgb_token = self.patchembd(rgb)
#         t_token = self.patchembd(t)
#         # fuse_token = torch.cat((rgb_token, t_token), 1)
#         fuse_token = rgb_token + t_token

#         for layer in self.layers:
#             fuse_out = layer(fuse_token)
#             # fuse_out += fuse_token
#             fuse = fuse_out.permute(0, 3, 1, 2).contiguous()
#             fuse_list.append(fuse)
#             fuse_token = fuse_out
#         neckout1, neckout2, neckout3 = self.neck_block(fuse_list[1], fuse_list[2], fuse_list[3])
#         res = self.detect_block([neckout1, neckout2, neckout3])
#         return res


class rgbt_replknet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False,
                 small_kernel_merged=False,
                 use_sync_bn=True,
                 norm_intermediate_features=False,       # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 training = False
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        # elif num_classes is not None and out_indices is not None:
        #     raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()
        self.gd = 1.0
        self.gw = 1.0
        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.rgb_stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        self.t_stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.rgb_stages = nn.ModuleList()
        self.t_stages = nn.ModuleList()
        self.fuse_stages = nn.ModuleList()
        
        self.rgb_transitions = nn.ModuleList()
        self.t_transitions = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.rgb_stages.append(layer)
            self.t_stages.append(layer)
            self.fuse_stages.append(fuse_block_conv1x1(ch=channels[stage_idx]))

            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.rgb_transitions.append(transition)
                self.t_transitions.append(transition)

        # if num_classes is not None:
        #     self.norm = get_bn(channels[-1])
        #     self.avgpool = nn.AdaptiveAvgPool2d(1)
        #     self.head = nn.Linear(channels[-1], num_classes)\
        
        self.neck_block = Yolov5Neck(channels[-1], n=3, gd=self.gd, gw=self.gw, last_ch=channels[-1])
        self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        self.detect_block = Yolov5DetectHead(num_classes, self.anchors, ch=[int(channels[-1]/4*self.gw), int(channels[-1]/2*self.gw), int(channels[-1]*self.gw)], training=training)

    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    def forward_features(self, rgb, t):
        rgb = self.rgb_stem[0](rgb)
        t = self.t_stem[0](t)

        for stem_layer in self.rgb_stem[1:]:
            if self.use_checkpoint:
                rgb = checkpoint.checkpoint(stem_layer, rgb)     # save memory
            else:
                rgb = stem_layer(rgb)
        
        for stem_layer in self.t_stem[1:]:
            if self.use_checkpoint:
                t = checkpoint.checkpoint(stem_layer, t)     # save memory
            else:
                t = stem_layer(t)

        if self.out_indices is None:
            #   Just need the final output
            # for stage_idx in range(self.num_stages):
            #     x = self.stages[stage_idx](x)
            #     if stage_idx < self.num_stages - 1:
            #         x = self.transitions[stage_idx](x)
            # return x
            raise ValueError('out_indices is None!')
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                rgb = self.rgb_stages[stage_idx](rgb)
                t = self.t_stages[stage_idx](t)

                if stage_idx in self.out_indices:
                    rgb = self.rgb_stages[stage_idx].norm(rgb)
                    t = self.t_stages[stage_idx].norm(t)

                    fuse = self.fuse_stages[stage_idx](rgb, t)
                    rgb = rgb + fuse
                    t = t + fuse

                    outs.append(fuse)     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    rgb = self.rgb_transitions[stage_idx](rgb)
                    t = self.t_transitions[stage_idx](t)

            return outs

    def forward(self, rgb, t):
        outs = self.forward_features(rgb, t)
        if self.out_indices:
            neckout1, neckout2, neckout3 = self.neck_block(outs[0], outs[1], outs[2])
            res = self.detect_block([neckout1, neckout2, neckout3])
            return res
        else:
            # x = self.norm(x)
            # x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            # x = self.head(x)
            # return x
            raise ValueError('out_indices is None!')
        
    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()

default_UniRepLKNet_A_F_P_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_N_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_T_kernel_sizes = ((3, 3, 3),
                                      (13, 13, 13),
                                      (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                      (13, 13, 13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = ((3, 3, 3),
                                             (13, 13, 13),
                                             (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                             (13, 13, 13))
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes
}

class rgbt_unireplknet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=(3, 3, 27, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 kernel_sizes=None,
                 deploy=False,
                 with_cp=False,
                #  init_cfg=None,
                 attempt_use_lk_impl=True,
                 use_sync_bn=False,
                 training = True,
                 **kwargs
                 ):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths, '
                                 'please specify kernel sizes for each block, e.g., '
                                 '((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))')
        print(kernel_sizes)
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.gd = 1.0
        self.gw = 1.0
        self.with_cp = with_cp

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print('=========== drop path rates: ', dp_rates)

        self.rgb_downsample_layers = nn.ModuleList()
        self.t_downsample_layers = nn.ModuleList()
        self.fuse_stages = nn.ModuleList()

        self.rgb_downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        
        self.t_downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))

        for i in range(3):
            self.rgb_downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))
            
            self.t_downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))
            self.fuse_stages.append(fuse_block_conv1x1(dims[i+1]))

        self.rgb_stages = nn.ModuleList()
        self.t_stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[UniRepLKNetBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn) for j in
                  range(depths[i])])
            self.rgb_stages.append(main_stage)
            self.t_stages.append(main_stage)
            
            cur += depths[i]
        
        last_channels = dims[-1]

        self.neck_block = Yolov5Neck(last_channels, n=3, gd=self.gd, gw=self.gw, last_ch=last_channels)
        self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        self.detect_block = Yolov5DetectHead(num_classes, self.anchors, ch=[int(last_channels/4*self.gw), int(last_channels/2*self.gw), int(last_channels*self.gw)], training=training)

        # self.for_pretrain = init_cfg is None
        # self.for_downstream = not self.for_pretrain     # there may be some other scenarios
        # if self.for_downstream:
        #     assert num_classes is None

        # if self.for_pretrain:
        #     self.init_cfg = None
        #     self.norm = nn.LayerNorm(last_channels, eps=1e-6)  # final norm layer
        #     self.head = nn.Linear(last_channels, num_classes)
        #     self.apply(self._init_weights)
        #     self.head.weight.data.mul_(head_init_scale)
        #     self.head.bias.data.mul_(head_init_scale)
        #     self.output_mode = 'logits'
        # else:
        self.init_cfg = None        # OpenMMLab style init
        # self.init_weights()
        self.output_mode = 'features'
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def gw_div(self, x):
        divisor = 8 
        x *= self.gw
        return int(math.ceil(x / divisor) * divisor)
    
    #   load pretrained backbone weights in the OpenMMLab style
    def init_weights(self):

        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            load_state_dict(self, _state_dict, strict=False, logger=logger)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, t):
        # if self.output_mode == 'logits':
        #     for stage_idx in range(4):
        #         x = self.downsample_layers[stage_idx](x)
        #         x = self.stages[stage_idx](x)
        #     x = self.norm(x.mean([-2, -1]))
        #     x = self.head(x)
        #     return x
        # el
        if self.output_mode == 'features':
            outs = []
            for stage_idx in range(4):
                rgb = self.rgb_downsample_layers[stage_idx](rgb)
                t = self.t_downsample_layers[stage_idx](t)
                
                rgb = self.rgb_stages[stage_idx](rgb)
                t = self.t_stages[stage_idx](t)
                if stage_idx > 0:
                    fuse = self.fuse_stages[stage_idx-1](rgb, t)
                    rgb += fuse
                    t += fuse
                    outs.append(self.__getattr__(f'norm{stage_idx}')(fuse))
            neckout1, neckout2, neckout3 = self.neck_block(outs[0], outs[1], outs[2])
            res = self.detect_block([neckout1, neckout2, neckout3])
            return res
        else:
            raise ValueError('Defined new output mode?')

    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
    

class rgbt_RTDETR(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0, last_ch=1024, nc=6, training=False) -> None:
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
    
        self.neck_block = RTDETRNeck(last_ch, n=3, gd=self.gd, gw=self.gw, last_ch=last_ch)
        self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        self.detect_block = Yolov5DetectHead(nc, self.anchors, ch=[int(last_ch/4*self.gw), int(last_ch/4*self.gw), int(last_ch/4*self.gw)], training=training)

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

def rgbtmodel_factory(model_name='rgbt_yolov5',ch=3, nc=1, gd=0.33, gw=0.5, training=True) -> nn.modules :
    if model_name == 'rgbt_yolov5':
        return rgbt_yolov5(ch=ch, nc=nc, gd=gd, gw=gw, training=True)
    elif model_name == 'rgbt_yolov5_aff':
        return rgbt_yolov5_AFF(ch=ch, nc=nc, gd=gd, gw=gw, training=True)
    elif model_name == 'rgbt_yolov5_ca':
        return rgbt_yolov5_CA(ch=ch, nc=nc, gd=gd, gw=gw, training=True)
    elif model_name == 'rgbt_replknet':
        return rgbt_replknet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], 
                             channels=[128,256,512,1024], drop_path_rate=0.3, small_kernel=5, 
                             num_classes=nc, use_checkpoint=False, small_kernel_merged=False, 
                             out_indices=[1,2,3], training= True)
    elif model_name == 'rgbt_unireplknet':
        return rgbt_unireplknet(ch, num_classes=nc, dims=(128, 256, 512, 1024), depths=(2, 2, 8, 2), attempt_use_lk_impl=True)
    elif model_name == 'rgbt_rtdetr':
        return rgbt_RTDETR(ch=ch, nc=nc, gd=gd, gw=gw, training=True)
    else :
        raise ValueError("unsupported model_name!:{}".format(model_name))


    



