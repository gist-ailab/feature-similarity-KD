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



# Cross Sampling = True (DDP = True, MODE=IR)
MARGIN=CosFace
F_M=0.2
for SIZE_TYPE in range fix
do
    for LR_PROB in 0.2 1.0
    do
        for CHOICE in 1 0
        do 
            if [ "$CHOICE" = "0" ]; then
                PHOTO_PROB=0.0
            elif [ "$CHOICE" = "1" ]; then
                PHOTO_PROB=$LR_PROB
            fi

            SEED=5
            BACKBONE=iresnet50
            METHOD=F_SKD_CROSS_BN
            PARAM=20.0,4.0
            CMARGIN=0.0
            RESOLUTION=1
            POOLING=E
            DATASET=webface4m
            TEACHER=checkpoint/teacher-webface4m/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
            CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=51 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                            --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                            --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                            --save_dir checkpoint/student-$DATASET/ablation_augparam/$BACKBONE-$MARGIN/F_M{$F_M}-size{$SIZE_TYPE}-photo{$PHOTO_PROB}-lr{$LR_PROB} \
                                                                            --batch_size 512 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True --margin_float $F_M \
                                                                            --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB
        done
    done
done