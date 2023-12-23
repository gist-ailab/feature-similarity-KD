## Teacher Training
for MARGIN in AdaFace
do
    SEED=5

    # Teacher Train
    # Set MARGIN_F based on the MARGIN value
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.4
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi

    DATASET=webface4m
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=621 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $MARGIN_F
done



## Naive Test
MARGIN=AdaFace
SIZE_TYPE=fix
for LR_PROB in 0.2 1.0
do
    for CHOICE in 1 0
    do 
        if [ "$CHOICE" = "0" ]; then
            PHOTO_PROB=0.0
        elif [ "$CHOICE" = "1" ]; then
            PHOTO_PROB=$LR_PROB
        fi
        F_M=0.4
        SEED=5
        DATASET=webface4m
        RESOLUTION=1
        POOLING=E
        BACKBONE=iresnet50
        CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=89 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/ablation_augparam/$BACKBONE-$MARGIN/F_M{$F_M}-size{$SIZE_TYPE}-photo{$PHOTO_PROB}-lr{$LR_PROB} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $F_M \
                                --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB
    done
done