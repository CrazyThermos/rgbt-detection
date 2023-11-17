import math
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from common import Conv, SPPF, Yolov5Neck
from PIL import Image
from matplotlib import pyplot as plt
from backbone import *
from fuselayer import *

class layer_fusion_1(nn.Module):
    def __init__(self, ch, gd=1.0, gw=1.0) -> None:
        super().__init__()
        self.gd = gd
        self.gw = gw
        self.rgb_conv_1 = Conv(ch, self.gw_div(64), 6, 2, 2)
        self.rgb_block1 = yolov5_backbone_block(64, 128, 3, gd=gd, gw=gw)
        self.rgb_block2 = yolov5_backbone_block(128, 256, 6, gd=gd, gw=gw)
        self.rgb_block3 = yolov5_backbone_block(256, 512, 9, gd=gd, gw=gw)
        self.rgb_block4 = yolov5_backbone_block(512, 1024, 3, gd=gd, gw=gw)
        self.rgb_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.t_conv_1 = Conv(ch, self.gw_div(64), 6, 2, 2)
        self.t_block1 = yolov5_backbone_block(64, 128, 3, gd=gd, gw=gw)
        self.t_block2 = yolov5_backbone_block(128, 256, 6, gd=gd, gw=gw)
        self.t_block3 = yolov5_backbone_block(256, 512, 9, gd=gd, gw=gw)
        self.t_block4 = yolov5_backbone_block(512, 1024, 3, gd=gd, gw=gw)
        # self.t_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.fuse_block1 = fuse_block_conv1x1(self.gw_div(128))
        self.fuse_block2 = fuse_block_conv1x1(self.gw_div(256))
        self.fuse_block3 = fuse_block_conv1x1(self.gw_div(512))
        self.fuse_block4 = fuse_block_conv1x1(self.gw_div(1024))
    
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
        self.gd = gd
        self.gw = gw
        self.rgb_conv_1 = Conv(ch, self.gw_div(64), 6, 2, 2)
        self.rgb_block1 = yolov5_backbone_block(64, 128, 3, gd=gd, gw=gw)
        self.rgb_block2 = yolov5_backbone_block(128, 256, 6, gd=gd, gw=gw)
        self.rgb_block3 = yolov5_backbone_block(256, 512, 9, gd=gd, gw=gw)
        self.rgb_block4 = yolov5_backbone_block(512, 1024, 3, gd=gd, gw=gw)
        self.rgb_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.t_conv_1 = Conv(ch, self.gw_div(64), 6, 2, 2)
        self.t_block1 = yolov5_backbone_block(64, 128, 3, gd=gd, gw=gw)
        self.t_block2 = yolov5_backbone_block(128, 256, 6, gd=gd, gw=gw)
        self.t_block3 = yolov5_backbone_block(256, 512, 9, gd=gd, gw=gw)
        self.t_block4 = yolov5_backbone_block(512, 1024, 3, gd=gd, gw=gw)
        # self.t_sppf   = SPPF(self.gw_div(1024), self.gw_div(1024), 5)

        self.fuse_block1 = fuse_block_conv1x1(128)
        self.fuse_block2 = fuse_block_conv1x1(256)
        self.fuse_block3 = fuse_block_conv1x1(512)
        self.fuse_block4 = fuse_block_conv1x1(1024)
    
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


def create_model():
    pass


def dump_feature_map(layer, prefix, feature):
    for i in layer:
        plt.imsave("feature_"+prefix+"_"+str(layer)+"_out.png", feature[i][0].transpose(0,1).sum(1).detach().numpy())
    


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize(640),
                                    transforms.CenterCrop(640),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # rgb = cv.imread("./test_imgs/1686403336-rgb.png")
    # rgb_tensor = torch.Tensor(rgb).unsqueeze(0).permute(0,3,1,2)
    # t = cv.imread("./test_imgs/1686403336-t.png")
    # t_tensor = torch.Tensor(t).unsqueeze(0).permute(0,3,1,2)

    rgb = Image.open("../test_imgs/1686403336-rgb.png")
    t   = Image.open("../test_imgs/1686403336-t.png")
    rgb = transform(rgb).unsqueeze(0)
    t   = transform(t).unsqueeze(0)

    cfg = "yolov5s"
    layer_yolov5s = ["conv_1.act_32", "c3_1.cv3.bn", "c3_2.cv3.bn", "c3_3.cv3.bn", "c3_4.cv3.bn"]
    layer_yolov5l = ["conv_1.act_60", "c3_1.cv3.bn", "c3_2.cv3.bn", "c3_3.cv3.bn", "c3_4.cv3.bn"]

    backbone = None
    layer = None
    if cfg == "yolov5s":   
        backbone = yolov5_backbone(3,gd=0.33,gw=0.5)
        print(backbone)
        nodes, _ = get_graph_node_names(backbone)
        # print(nodes)
        layer = layer_yolov5s
    elif cfg == "yolov5l":
        backbone = yolov5_backbone(3,gd=1.0,gw=1.0)
        print(backbone)
        nodes, _ = get_graph_node_names(backbone)
        # print(nodes)
        layer = layer_yolov5l

    # feature_rgb1024 = backbone(rgb_tensor)
    # feature_t1024 = backbone(t_tensor)

    # feature_extractor = create_feature_extractor(backbone, return_nodes=layer)
    # feature_rgb1024 = feature_extractor(rgb)
    # feature_t1024 = feature_extractor(t)

    # plt.imsave("feature_rgb_out.png", feature_rgb1024[layer[0]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_t_out.png", feature_t1024[layer[0]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_rgb128.png", feature_rgb1024[layer[1]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_t128.png", feature_t1024[layer[1]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_rgb256.png", feature_rgb1024[layer[2]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_t256.png", feature_t1024[layer[2]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_rgb512.png", feature_rgb1024[layer[3]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_t512.png", feature_t1024[layer[3]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_rgb1024.png", feature_rgb1024[layer[4]][0].transpose(0,1).sum(1).detach().numpy())
    # plt.imsave("feature_t1024.png", feature_t1024[layer[4]][0].transpose(0,1).sum(1).detach().numpy())


    fusion = layer_fusion_1(3,gd=0.33,gw=0.5)
    output = fusion(rgb, t)#[0].transpose(0,1).sum(1).detach().numpy()
    # plt.imsave("feature_fuse1.png", output[0])
    # plt.imsave("feature_fuse2.png", output[1])
    # plt.imsave("feature_fuse3.png", output[2])

    neck = Yolov5Neck(1024,gd=0.33,gw=0.5)
    print(neck)
    output2 = neck(output[0], output[1], output[2])
    print(output2)