################################## WebFace ###############################
MARGIN=AdaFace
SIZE_TYPE=range
LR_PROB=0.2
PHOTO_PROB=0.2
F_M=0.4
SEED=1
DATASET=webface4m
RESOLUTION=1
POOLING=E
BACKBONE=iresnet50
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=43 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case1/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB



MARGIN=AdaFace
SIZE_TYPE=fix
LR_PROB=0.2
PHOTO_PROB=0.2
F_M=0.4
SEED=1
DATASET=webface4m
RESOLUTION=1
POOLING=E
BACKBONE=iresnet50
CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=41 train_teacher_multi.py --seed $SEED --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/final_ablation/$DATASET/case1/HR-LR-PHOTO{$PHOTO_PROB},LR{$LR_PROB},type{$SIZE_TYPE}/$BACKBONE-$MARGIN-$F_M \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $F_M \
                        --size_type $SIZE_TYPE --lr_prob $LR_PROB --photo_prob $PHOTO_PROB