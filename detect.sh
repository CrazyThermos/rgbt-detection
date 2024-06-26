# python detect.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/rgbt-aff-med/weights/best.pt' \
#                 --conf-thres 0.25 \
#                 --data 'dataset/test.yaml' \
#                 --imgsz 1280  \
#                 --device 0 \
#                 --source '/home/zhengyuhang/datasets/TEST/images/val' \
#                 --model-name 'rgbt_yolov5_aff'
                # --draw_edge

python detect.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/rtdetr_516/weights/best.pt' \
                --conf-thres 0.25 \
                --data 'dataset/result.yaml' \
                --imgsz 640  \
                --device 0 \
                --source '/home/zhengyuhang/datasets/result516_yolo/images/val' \
                --model-name 'rgbt_rtdetr'