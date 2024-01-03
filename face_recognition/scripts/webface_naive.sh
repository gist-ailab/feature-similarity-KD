# Teacher - Naive - F-SKD
S=64.0
DATASET=webface4m
SEED=5
POOLING=E                 
DATA_DIR=/SSDb/sung/dataset/face_dset/

for MARGIN in ArcFace
do
    for MP in True
    do
        if [ "$MARGIN" = "CosFace" ]; then
            M_F=0.4
        elif [ "$MARGIN" = "ArcFace" ]; then
            M_F=0.5
        elif [ "$MARGIN" = "AdaFace" ]; then
            M_F=0.4
        fi
        RESOLUTION=1
        CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=161 train_teacher_multi.py \
                                --seed $SEED --data_dir $DATA_DIR --save_dir checkpoint/naive-$DATASET/iresnet50-$MARGIN-m{$M_F}-s{$S}-MP{$MP} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone iresnet50 --dataset $DATASET --batch_size 512 \
                                --scale $S --margin_float $M_F --mixed_precision $MP
    done
done
