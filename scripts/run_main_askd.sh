# Teacher Training
for MARGIN in ArcFace CosFace AdaFace
do
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed 5 --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher/$BACKBONE-$POOLING-CBAM-$MARGIN/ \
                            --down_size $RESOLUTION --mode cbam --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE
done


# Student Training
for MARGIN in ArcFace CosFace AdaFace
do
    for BACKBONE in iresnet50
    do
        METHOD=A_SKD
        PARAM=80.0,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-CBAM-$MARGIN/last_net.ckpt
        python train_student.py --seed 5 --gpus 1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode cbam --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-CBAM-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}
    done
done