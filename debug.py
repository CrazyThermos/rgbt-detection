from model.frame import *
from model.neck import Yolov5Neck
from model.backbone import *
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torch
import yaml
import pandas as pd
import matplotlib
from dataset.rgbt_dataset import create_rgbtdataloader
from utils.general import LOCAL_RANK
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640

def create_model():
    pass


def dump_feature_map(layer, prefix, feature):
    for i in layer:
        plt.imsave("feature_"+prefix+"_"+str(layer)+"_out.png", feature[i][0].transpose(0,1).sum(1).detach().numpy())


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                    transforms.CenterCrop(IMG_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # rgb = cv.imread("./test_imgs/1686403336-rgb.png")
    # rgb_tensor = torch.Tensor(rgb).unsqueeze(0).permute(0,3,1,2)
    # t = cv.imread("./test_imgs/1686403336-t.png")
    # t_tensor = torch.Tensor(t).unsqueeze(0).permute(0,3,1,2)

    rgb = Image.open("test_imgs/010001.jpg")
    t   = Image.open("test_imgs/010001_ir.jpg")
    rgb = transform(rgb).unsqueeze(0).to(device)
    t   = transform(t).unsqueeze(0).to(device)

    cfg = "yolov5s"
    layer_yolov5s = ["conv_1.act_32", "c3_1.cv3.bn", "c3_2.cv3.bn", "c3_3.cv3.bn", "c3_4.cv3.bn"]
    layer_yolov5l = ["conv_1.act_60", "c3_1.cv3.bn", "c3_2.cv3.bn", "c3_3.cv3.bn", "c3_4.cv3.bn"]

    backbone = None
    layer = None
    
    # if cfg == "yolov5s":   
    #     backbone = yolov5_backbone(3,gd=0.33,gw=0.5)
    #     print(backbone)
    #     nodes, _ = get_graph_node_names(backbone)
    #     # print(nodes)
    #     layer = layer_yolov5s
    # elif cfg == "yolov5l":
    #     backbone = yolov5_backbone(3,gd=1.0,gw=1.0)
    #     print(backbone)
    #     nodes, _ = get_graph_node_names(backbone)
    #     # print(nodes)
    #     layer = layer_yolov5l

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


    # fusion = RGBTModel(3)
    # output = fusion(rgb, t)#[0].transpose(0,1).sum(1).detach().numpy()
    # print(output[0][0][0][0])
    # print(output[0][0][0][1])
    # print(output[0][0][0][2])
    # print(output[0][0][0][3])
    # print(output[0][0][0][4])
    # print(output[0][0][0][5])
    # print(output[0][0][0][6])

    # plt.imsave("feature_fuse1.png", output[0])
    # plt.imsave("feature_fuse2.png", output[1])
    # plt.imsave("feature_fuse3.png", output[2])

    # neck = Yolov5Neck(1024,gd=0.33,gw=0.5)
    # print(neck)
    # output2 = neck(output[0], output[1], output[2])
    # print(output2[0].shape)
    # print(output2[1].shape)
    # print(output2[2].shape)



    # hyp = "configs/hyp.scaratch-low.yaml"
    # with open(hyp, errors='ignore') as f:
    #     hyp = yaml.safe_load(f)  # load hyps dict
    #     if 'anchors' not in hyp:  # anchors commented in hyp.yaml
    #         hyp['anchors'] = 3
    # ndataloader, ndataset = create_rgbtdataloader(path="../../datasets/TEST",
    #                                         imgsz=640,
    #                                         batch_size=1,
    #                                         stride=32,
    #                                         single_cls=False,
    #                                         hyp=hyp,
    #                                         augment=False,
    #                                         cache="ram",
    #                                         rect=False,
    #                                         rank=LOCAL_RANK,
    #                                         workers=8,
    #                                         image_weights=False,
    #                                         quad=False,
    #                                         prefix='train:',
    #                                         shuffle=True,
    #                                         seed=0)
    # pbar = enumerate(ndataloader)
    # for i, (img_rgb, img_t, targets, paths, _) in pbar:  
    #     print(img_rgb)
    #     print(img_t)
    #     pass


    '''
    debug train_loader
    '''
    # import tqdm
    # val_loader, dataset = create_rgbtdataloader('/home/cv/Project1/yuhang/datasets/TEST/images/val',
    #                                         IMG_SIZE,
    #                                         4,
    #                                         stride=32,
    #                                         single_cls=False,
    #                                         hyp='/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/configs/hyp.scratch-low.yaml',
    #                                         rect=True,
    #                                         workers=8,
    #                                         rank=-1,
    #                                         pad=0.0,
    #                                         prefix='val: '
    #                                         )
    # import val as validate
    # from utils.loss import ComputeLoss 
    # from utils.callbacks import Callbacks
    # from utils.general import check_dataset

    # nc=1
    # imgsz=IMG_SIZE
    # nl=3
    # hyp='configs/hyp.scratch-low.yaml'
    # with open(hyp, errors='ignore') as f:
    #         hyp = yaml.safe_load(f)
    # hyp['box'] *= 3 / nl  # scale to layers
    # hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # backendmodel = RGBTModel(3, nc=1, gd=0.33,gw=0.5).to(device='cuda')
    # backendmodel.names = ['person']
    # backendmodel.hyp = hyp
    # data_dict = check_dataset('dataset/test.yaml') 
    # callbacks = Callbacks()
    # results, _, _ = validate.run(data_dict,
    #                              backendmodel=backendmodel,
    #                             weights='/home/cv/Project1/yuhang/RGBT-Detection/runs/train/rgbt_yolov5s_llvip_693/weights/best.pt',
    #                             batch_size=1,
    #                             imgsz=IMG_SIZE,
    #                             model=None, # attempt_load(f, device).half(),
    #                             iou_thres=0.60,  # best pycocotools at iou 0.65
    #                             single_cls=True,
    #                             dataloader=val_loader,
    #                             save_dir='./output/backend',
    #                             save_json=False,
    #                             save_txt=True,
    #                             save_conf=True,
    #                             verbose=True,
    #                             plots=True,
    #                             callbacks=callbacks,
    #                             compute_loss=ComputeLoss(backendmodel))  # val best model with plots
    
    
    
    
    import numpy as np
    from model.mamba import PatchEmbed, Mamba, MambaConfig
    B, L, D, N = 16, 64, 768, 16
    testPatchEmbed = PatchEmbed(patch_size=4, embed_dim=96, img_size=IMG_SIZE).to(device)
    rgb_out = testPatchEmbed(rgb)
    t_out = testPatchEmbed(t)
    fuse_out = torch.cat((rgb_out,t_out),1).to(device)

    # with torch.no_grad():
    # config = MambaConfig(d_model=D, n_layers=8, d_state=N)
    # model = Mamba(config)
    model = rgbt_Mamba(ch=3).to(device)

    # rgb_mamba_out = model(rgb_out)
    # rgb_mamba_out = rgb_mamba_out.unsqueeze(3)
    # rgb_mamba_out = rgb_mamba_out.transpose(0, 3)
    # rgb_mamba_out = rgb_mamba_out.reshape(1, 3, IMG_SIZE, IMG_SIZE)
    # rgb_mamba_out = rgb_mamba_out + rgb 
    # rgb_mamba_out = rgb_mamba_out.squeeze(0)

    # t_mamba_out = model(t_out)
    # t_mamba_out = t_mamba_out.unsqueeze(3)
    # t_mamba_out = t_mamba_out.transpose(0, 3)
    # t_mamba_out = t_mamba_out.reshape(1, 3, IMG_SIZE, IMG_SIZE)
    # t_mamba_out = t_mamba_out + t + t
    # t_mamba_out = t_mamba_out.squeeze(0)

    fuse_mamba_out = model(rgb, t)
    fuse_mamba_out = fuse_mamba_out.unsqueeze(3)
    fuse_mamba_out = fuse_mamba_out.transpose(0, 3)
    fuse_mamba_out = fuse_mamba_out.reshape(2, 3, IMG_SIZE, IMG_SIZE)
    rgb_fuse = fuse_mamba_out[1] + t 
    
    # t_mamba_out = t_mamba_out + t
    # t_mamba_out = t_mamba_out.squeeze(0)
    # mamba_out = rgb_mamba_out + t_mamba_out + rgb.squeeze(0)
    
    p = transforms.ToPILImage(mode="RGB")
    rgb_im = p(rgb_fuse.squeeze(0))
    rgb_im.save('./mamba_fuse3_out.png')
    # p = transforms.ToPILImage(mode="RGB")
    # rgb_im = p(t_mamba_out)
    # rgb_im.save('./mamba_fuse4_out.png')

    # from model.frame import *
    # x = torch.randn(2, 4, 19, 19)
    layer = rgbt_unireplknet(3, num_classes=6, dims=(128, 256, 512, 1024), depths=(3, 3, 18, 3), attempt_use_lk_impl=True).to(device)
    # for n, p in layer.named_parameters():
    #     if 'beta' in n:
    #         torch.nn.init.ones_(p)
    #     else:
    #         torch.nn.init.normal_(p)
    # for n, p in layer.named_buffers():
    #     if 'running_var' in n:
    #         print('random init var')
    #         torch.nn.init.uniform_(p)
    #         p.data += 2
    #     elif 'running_mean' in n:
    #         print('random init mean')
    #         torch.nn.init.uniform_(p)
    # layer.gamma.data += 0.5
    layer.eval()
    rgb = rgb.to(device)
    t = t.to(device)
    origin_y = layer(rgb, t)
    layer.reparameterize_unireplknet()
    eq_y = layer(rgb, t)

    print("output difference:", ((origin_y - eq_y) ** 2).mean())