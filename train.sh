 #!/usr/bin/env
python train.py --data dataset/m3fd.yaml \
                --hyp configs/hyp.scratch-low.yaml \
                --model-name 'rgbt_rtdetr' \
                --optimizer SGD \
                --batch-size 32 \
                --epochs 330 \
                --img 640 \
                --name rtdetr \
                --device 2


# python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data dataset/m3fd.yaml \
#                 --hyp configs/hyp.scratch-low.yaml \
#                 --model-name 'rgbt_rtdetr' \
#                 --optimizer SGD \
#                 --batch-size 64 \
#                 --epochs 330 \
#                 --img 640 \
#                 --name rtdetr