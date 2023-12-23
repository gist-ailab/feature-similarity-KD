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
    INTERPOLATION=random
    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=621 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $MARGIN_F
done



## Teacher Training
for MARGIN in CosFace
do
    F_M=0.2
    SEED=5
    DATASET=webface4m
    RESOLUTION=1
    POOLING=E
    BACKBONE=iresnet50
    INTERPOLATION=random
    CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=911 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $F_M
    
done