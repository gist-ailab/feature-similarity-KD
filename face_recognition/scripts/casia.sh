# CASIA
S=32.0
DATASET=casia
SEED=5
POOLING=E                 
DATA_DIR=/home/jovyan/SSDb/sung/dataset/face_dset/

for MARGIN in CosFace
do
    for MP in False
    do
        if [ "$MARGIN" = "CosFace" ]; then
            M_F=0.4
        elif [ "$MARGIN" = "ArcFace" ]; then
            M_F=0.5
        elif [ "$MARGIN" = "AdaFace" ]; then
            M_F=0.4
        fi
        RESOLUTION=0
        python train_teacher.py --gpus 2 --seed $SEED --data_dir $DATA_DIR --save_dir checkpoint/teacher-$DATASET/iresnet50-$MARGIN-m{$M_F}-s{$S}-MP{$MP} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone iresnet50 --dataset $DATASET --batch_size 256 \
                                --scale $S --margin_float $M_F --mixed_precision $MP


        # Embedding
        if [ "$MARGIN" = "CosFace" ]; then
            M_F=0.4
        elif [ "$MARGIN" = "ArcFace" ]; then
            M_F=0.5
        elif [ "$MARGIN" = "AdaFace" ]; then
            M_F=0.4
        fi
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$MARGIN-m{$M_F}-s{$S}-MP{$MP}/last_net.ckpt
        python train_embed.py --gpus 2 --dataset $DATASET --data_dir $DATA_DIR --mode ir --seed $SEED --teacher_path $TEACHER
    

        # Student
        if [ "$MARGIN" = "CosFace" ]; then
            M_F=0.4
        elif [ "$MARGIN" = "ArcFace" ]; then
            M_F=0.5
        elif [ "$MARGIN" = "AdaFace" ]; then
            M_F=0.4
        fi
        RESOLUTION=1                
        BACKBONE=iresnet18
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$MARGIN-m{$M_F}-s{$S}-MP{$MP}/last_net.ckpt
        SAVE_DIR=checkpoint/student-$DATASET/$BACKBONE-$MARGIN-m{0.2}-s{$S}-MP{$MP}/seed{$SEED}
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=311 train_student_multi.py --seed $SEED --data_dir $DATA_DIR --down_size $RESOLUTION \
                                                                    --backbone $BACKBONE --mode ir --margin_type $MARGIN --pooling $POOLING \
                                                                    --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                    --save_dir $SAVE_DIR --batch_size 256 --dataset $DATASET --cross_margin 0.0 \
                                                                    --cross_sampling True --hint_bn True --margin_float 0.2 --scale $S \
                                                                    --mixed_precision $MP
    done
done
