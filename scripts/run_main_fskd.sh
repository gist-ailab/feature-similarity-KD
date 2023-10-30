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


# Student Training
for MARGIN in ArcFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        METHOD=F_SKD_CROSS
        PARAM=20.0,4.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=ms1mv2
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
        python train_student.py --seed 5 --gpus 2,3 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM} \
                                --batch_size 512 --dataset $DATASET
    done
done