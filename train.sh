python train.py --data dataset/llvip.yaml \
                --hyp configs/hyp.scratch-med.yaml \
                --optimizer SGD \
                --batch-size 8 \
                --epochs 300 \
                --img 1280 \
                --name rgbt \
                --device 1