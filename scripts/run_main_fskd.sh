# Teacher Training
for MARGIN in MagFace
do
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed 5 --gpus 0 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/ \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256
done


# Naive Training
for MARGIN in MagFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        DATASET=casia
        RESOLUTION=1
        POOLING=E
        INTERPOLATION=random
        python train_teacher.py --seed 5 --gpus 1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION \
                                --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 256
    done
done


# Student Training - w/ Cross-Sampling
for CMARGIN in 0.5 0.6
do
    MARGIN=CosFace
    BACKBONE=iresnet50
    METHOD=F_SKD_CROSS_BN
    PARAM=20.0,4.0
    RESOLUTION=1
    INTERPOLATION=random
    POOLING=E
    DATASET=casia
    TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
    python train_student.py --seed 5 --gpus 0,1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                            --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                            --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN} \
                            --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True
done



# Student Training - No Cross Sampling
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
python train_student.py --seed 5 --gpus 2 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM} \
                        --batch_size 256 --dataset $DATASET --cross_margin 0.0 --cross_sampling False