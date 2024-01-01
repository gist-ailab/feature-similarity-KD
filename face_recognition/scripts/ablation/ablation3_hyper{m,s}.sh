# CosFace
SEED=5               
for MARGIN in CosFace ArcFace
do
    for F_M in 0.2
    do
        for S in 32. 64.
        do
            DATASET=casia
            RESOLUTION=1                
            POOLING=E                 
            BACKBONE=iresnet50            
            METHOD=F_SKD_CROSS_BN
            PARAM=20.0,4.0
            TEACHER=checkpoint/teacher-$DATASET/$BACKBONE-$MARGIN/last_net.ckpt
            SAVE_DIR=checkpoint/final_ablation/$DATASET/case3/$BACKBONE-$MARGIN-m{$F_M}-s{$S}/
            CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=991 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                        --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                        --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                        --cross_sampling True --hint_bn True --margin_float $F_M --scale $S
        done
    done
done






# CosFace
SEED=5              
for MARGIN in AdaFace
do
    for F_M in 0.2
    do
        for S in 32. 64.
        do
            DATASET=casia
            RESOLUTION=1                
            POOLING=E                 
            BACKBONE=iresnet50            
            METHOD=F_SKD_CROSS_BN
            PARAM=20.0,4.0
            TEACHER=checkpoint/teacher-$DATASET/$BACKBONE-$MARGIN/last_net.ckpt
            SAVE_DIR=checkpoint/final_ablation/$DATASET/case3/$BACKBONE-$MARGIN-m{$F_M}-s{$S}/
            CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=961 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                        --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                        --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                        --cross_sampling True --hint_bn True --margin_float $F_M --scale $S
        done
    done
done
