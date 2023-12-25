```
- teacher - LR (batch=1024 / lr=0.1) 셋팅 → 1일
  - m은 adaface 논문 고정
  - HR only training
  - LR only training (2개 case - fix or range)
```

################################## ## HR_only Training ## #######################
# 1. WEBFACE4M
for MARGIN in AdaFace CosFace ArcFace
do
    SEED=1

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
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=621 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 1024 --margin_float $MARGIN_F
done


# 2. CASIA
for MARGIN in CosFace
do
    SEED=1

    # Teacher Train
    # Set MARGIN_F based on the MARGIN value
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.4
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --gpus 0 --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256 --margin_float $MARGIN_F
done