# rgbt-detection

## 简单通用的RGBT目标检测框架
* 支持单阶段模型 √
* 支持双阶段模型 x
* 支持anchor base √
* anchor free x
* 支持rgb-t双模态的目标检测 √
* 支持rgb-t模型的剪枝 √
* 支持rgb-t模型的量化 √

## 使用方法
在 frame.py backbone.py fuseblock.py neck.py head.py中定义你的模型

在 backbone.py 中定义主干网络，使用预训练主干可省略

在 fuseblock.py 中定义融合模块

在 neck.py和head.py中定义neck和head层

在 frame.py中整个网络

训练命令如下
```
python train.py --data dataset/llvip.yaml --hyp configs/hyp.scratch-low.yaml --optimizer SGD --batch-size 8 --epochs 300 --img 1280 --name rgbt --device 0
```

## 文件来源
* ~~utils/utils.py from pytorch~~
* utils/coco/coco_eval.py from pytorch
* utils/coco/coco_utils.py from pytorch
* utils/coco/transfroms.py from pytorch
* utils/coco/engine.py from pytorch

* utils/loss.py from yolov5
* utils/metrics.py from yolov5
* utils/anchor.py from yolov5
* utils/general.py from yolov5
* utils/torch_utils.py from yolov5
* utils/autobatch.py from yolov5
* utils/callbacks.py from yolov5
* train.py from yolov5
* val.py from yolov5
* dataset/base_dataset.py from yolov5
* dataset/rgbt_dataset.py from yolov5






