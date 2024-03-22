python train.py --data dataset/m3fd.yaml \
                --hyp configs/hyp.scratch-low.yaml \
                --optimizer SGD \
                --batch-size 4 \
                --epochs 300 \
                --img 640 \
                --name mamba \
                --device 1