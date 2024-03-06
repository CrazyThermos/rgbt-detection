python detect.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/rgbt-aff-med/weights/best.pt' \
                --conf-thres 0.25 \
                --data 'dataset/test.yaml' \
                --imgsz 1280  \
                --device 0 \
                --source '/home/zhengyuhang/datasets/TEST/images/val' 
                # --draw_edge