# fix / range -> 2가지
# lr_prob = 0.2, 0.5, 1.0 → 3가지
# photo_prob = 0.0, 0.2 → 2가지
# margin = CosFace / ArcFace → 2가지


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
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=401 train_teacher_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case2/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M/ \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB



