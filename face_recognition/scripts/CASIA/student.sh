# CASIA - FSKD
SEED=5               
for MARGIN in CosFace ArcFace AdaFace
do
    F_M=0.2
    S=64.0

    if [ "$MARGIN" = "CosFace" ]; then
        T_F_M=0.4
    elif [ "$MARGIN" = "ArcFace" ]; then
        T_F_M=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        T_F_M=0.4
    fi

    DATASET=casia
    RESOLUTION=1                
    POOLING=E                 
    BACKBONE=iresnet50            
    METHOD=F_SKD_CROSS_BN
    PARAM=20.0,4.0
    TEACHER=checkpoint/teacher-$DATASET/$BACKBONE-$MARGIN-m{$T_F_M}-s{$S}/last_net.ckpt
    SAVE_DIR=checkpoint/student-$DATASET/$BACKBONE-$MARGIN-m{$F_M}-s{$S}/seed{$SEED}
    CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=671 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                --cross_sampling True --hint_bn True --margin_float $F_M --scale $S
done