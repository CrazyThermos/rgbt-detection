from model.frame import RGBTModel

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
from model.common import attempt_load
import os

DEVICE = 'cpu'
dummy_input = torch.randn(1,3,1280,1280).to(device=DEVICE)
input_names=['input0','input1']
output_names=['output0']
weights='/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/best.pt'

model = RGBTModel(3, nc=1, gd=0.33,gw=0.5, training=False).eval()
model.detect_block.export = True
backendmodel = attempt_load(model, weights, DEVICE)
torch.onnx.export(backendmodel, (dummy_input, dummy_input),'rgbt_yolov5_op13.onnx', do_constant_folding=True, opset_version=13, verbose=True,input_names=input_names, output_names=output_names)