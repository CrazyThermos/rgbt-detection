 #!/usr/bin/env
python train.py --data dataset/m3fd.yaml \
                --hyp configs/hyp.scratch-low_v3.yaml \
                --model-name 'rgbt_ca_rtdetrv2' \
                --optimizer SGD \
                --batch-size 32 \
                --epochs 500 \
                --img 640 \
                --name rgbt_ca_rtdetrv2 \
                --device 3
                # --use-decoder \

# python train.py --data dataset/m3fd.yaml \
#                 --hyp configs/hyp.scratch-low.yaml \
#                 --model-name 'rgbt_yolov5_decoder' \
#                 --optimizer SGD \
#                 --batch-size 16 \
#                 --epochs 500 \
#                 --img 640 \
#                 --name rgbt_yolov5_decoder \
#                 --device 1 \
#                 --use-decoder
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data dataset/m3fd.yaml \
#                 --hyp configs/hyp.scratch-low.yaml \
#                 --model-name 'rgbt_rtdetr' \
#                 --optimizer SGD \
#                 --batch-size 64 \
#                 --epochs 330 \
#                 --img 640 \
#                 --name rtdetr