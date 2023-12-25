# fix / range -> 2가지
# lr_prob = 0.2, 0.5, 1.0 → 3가지
# photo_prob = 0.0, 0.2 → 2가지
# margin = CosFace / ArcFace → 2가지

# CosFace
MARGIN=CosFace         
F_M=0.4                  
SEED=1                     
for SIZE_TYPE in range fix
do
    for LR_PROB in 0.2 0.5 1.0
    do
        for PHOTO_PROB in 0.0 0.2
        do
            DATASET=casia
            RESOLUTION=1                
            POOLING=E                 
            BACKBONE=iresnet50            
            METHOD=F_SKD_CROSS_BN
            PARAM=20.0,4.0
            TEACHER=checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN/last_net.ckpt
            SAVE_DIR=checkpoint/final_ablation/$DATASET/case2/HR-LR-PHOTO{$PHOTO_PROB}-LR{$LR_PROB}-SIZE{$SIZE_TYPE}/$BACKBONE-$MARGIN-FSKD/
            CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=991 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                        --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                        --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                        --cross_sampling True --hint_bn True --margin_float $F_M \
                                                                        --lr_prob $LR_PROB --photo_prob $PHOTO_PROB --size_type $SIZE_TYPE
        done
    done
done




# CosFace
MARGIN=AdaFace         
F_M=0.4                  
SEED=1                     
for SIZE_TYPE in range fix
do
    for LR_PROB in 0.2 0.5 1.0
    do
        for PHOTO_PROB in 0.0 0.2
        do
            DATASET=casia
            RESOLUTION=1                
            POOLING=E                 
            BACKBONE=iresnet50            
            METHOD=F_SKD_CROSS_BN
            PARAM=20.0,4.0
            TEACHER=checkpoint/final_ablation/$DATASET/case0/HR_only/$BACKBONE-$MARGIN/last_net.ckpt
            SAVE_DIR=checkpoint/final_ablation/$DATASET/case2/HR-LR-PHOTO{$PHOTO_PROB}-LR{$LR_PROB}-SIZE{$SIZE_TYPE}/$BACKBONE-$MARGIN-FSKD/
            CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=911 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                        --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                        --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                        --cross_sampling True --hint_bn True --margin_float $F_M \
                                                                        --lr_prob $LR_PROB --photo_prob $PHOTO_PROB --size_type $SIZE_TYPE
        done
    done
done