# rgbt-detection

## 简单通用的RGBT目标检测框架
* 支持单阶段模型 √
* 支持双阶段模型 x
* 支持anchor base √
* anchor free x
* 支持rgb-t双模态的目标检测 √
* 支持rgb-t模型的剪枝 x
* 支持rgb-t模型的量化 √

## 训练
在 model/frame.py model/backbone.py model/fuseblock.py model/neck.py head.py中定义你的模型

在 model/backbone.py 中定义主干网络，使用预训练主干可省略;

在 model/fuseblock.py 中定义融合模块;

在 model/neck.py和head.py中定义neck和head层;

在 model/frame.py中定义整个网络.

训练命令如下
```
python train.py --data dataset/llvip.yaml --hyp configs/hyp.scratch-low.yaml --optimizer SGD --batch-size 8 --epochs 300 --img 1280 --name rgbt --device 0
```
## 量化
本仓库使用openppl的ppq工具对模型进行量化

量化命令如下
```
python quantizer.py
```
或者
```
python quantization.py
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






