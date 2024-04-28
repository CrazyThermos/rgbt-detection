# python val.py --weights './tools/rgbt_yolov5_op13_quantized.engine' \
#                 --data 'dataset/m3fd.yaml' \
#                 --imgsz 1280  \
#                 --device 0 

                
python val.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/redetr/weights/best.pt' \
                --data 'dataset/m3fd.yaml' \
                --imgsz 640 \
                --device 1 \
                --model-name 'rgbt_rtdetr'