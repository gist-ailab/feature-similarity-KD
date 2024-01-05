# Teacher - Naive - F-SKD
S=64.0
DATASET=webface4m
SEED=5
POOLING=E                 
DATA_DIR=/SSDb/sung/dataset/face_dset/

for MARGIN in CosFace ArcFace AdaFace
do
    for MP in True
    do
        RESOLUTION=1
        CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=991 train_teacher_multi.py \
                                --seed $SEED --data_dir $DATA_DIR --save_dir checkpoint/naive-$DATASET/iresnet50-$MARGIN-m{0.2}-s{$S}-MP{$MP} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone iresnet50 --dataset $DATASET --batch_size 512 \
                                --scale $S --margin_float 0.2 --mixed_precision $MP
    done
done
