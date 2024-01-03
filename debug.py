from model.frame import layer_fusion_1
from model.neck import Yolov5Neck
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

    rgb = Image.open("test_imgs/1686403336-rgb.png")
    t   = Image.open("test_imgs/1686403336-t.png")
    rgb = transform(rgb).unsqueeze(0)
    t   = transform(t).unsqueeze(0)

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


    # fusion = layer_fusion_1(3)
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
    import tqdm
    train_loader, dataset = create_rgbtdataloader('/home/zhengyuhang/datasets/TEST/images/train',
                                            640,
                                            2,
                                            stride=32,
                                            single_cls=False,
                                            hyp='/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/configs/hyp.scratch-low.yaml',
                                            augment=False,
                                            cache=False,
                                            rect=False,
                                            rank=LOCAL_RANK,
                                            workers=8,
                                            image_weights=False,
                                            quad=False,
                                            prefix='train: ',
                                            shuffle=True,
                                            seed=0)
    # train_loader.sampler.set_epoch(4)
    pbar = enumerate(train_loader)
    # pbar = tqdm(pbar) 
    for i, (img_rgb, img_t, targets, rgb_paths,t_paths, _) in pbar:
        print(rgb_paths)
        print(t_paths)
        print(targets)