```
- teacher - LR (batch=1024 / lr=0.1) 셋팅 → 1일
  - m은 adaface 논문 고정
  - HR only training
  - LR only training (2개 case - fix or range)
```

################################## ## HR_only Training ## #######################
# 1. WEBFACE4M
for MARGIN in AdaFace
do
    SEED=5

    # Teacher Train
    # Set MARGIN_F based on the MARGIN value
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.35
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi
    DATASET=webface4m
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=621 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 1024 --margin_float $MARGIN_F
done


# 2. CASIA
for MARGIN in CosFace AdaFace
do
    SEED=5

    # Teacher Train
    # Set MARGIN_F based on the MARGIN value
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.35
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=621 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $MARGIN_F
done



################################## ## HR-LR with fixed sampling ## #######################
# 1. WebFace4M
MARGIN=AdaFace
SIZE_TYPE=fix
LR_PROB=0.2
PHOTO_PROB=0.2
F_M=0.4
SEED=5
DATASET=webface4m
RESOLUTION=1
POOLING=E
BACKBONE=iresnet50
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=41 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M/ \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 1024 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB


# 2. CASIA
MARGIN=AdaFace
SIZE_TYPE=fix
LR_PROB=0.2
PHOTO_PROB=0.2
F_M=0.4
SEED=5
DATASET=casia
RESOLUTION=1
POOLING=E
BACKBONE=iresnet50
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=41 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M/ \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 1024 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB




################################## ## HR-LR with range sampling ## #######################
# 1. WebFace4M
MARGIN=AdaFace
SIZE_TYPE=range
LR_PROB=0.2
PHOTO_PROB=0.2
F_M=0.4
SEED=5
DATASET=webface4m
RESOLUTION=1
POOLING=E
BACKBONE=iresnet50
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=43 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 1024 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB

