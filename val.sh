# python val.py --weights './tools/rgbt_ca_rtdetrv2_589_m3fd_int8_SYMM_LINEAR_PERCHANNEL.engine' \
#                 --data 'dataset/m3fd.yaml' \
#                 --imgsz 640  \
#                 --device 0 \
#                 --model-name 'rgbt_ca_rtdetrv2'

python val.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/rgbt_ca_rtdetrv24_589/weights/best.pt' \
                --data 'dataset/m3fd.yaml' \
                --imgsz 640  \
                --device 0 \
                --model-name 'rgbt_ca_rtdetrv2'
                
# python val.py --weights '/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/runs/train/redetr/weights/best.pt' \
#                 --data 'dataset/m3fd.yaml' \
#                 --imgsz 640 \
#                 --device 1 \
#                 --model-name 'rgbt_rtdetr'