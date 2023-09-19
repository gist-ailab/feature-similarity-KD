# Student Training - AT
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        METHOD=AT
        PARAM=1000,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
        python train_student.py --seed 5 --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}
    done
done


# Student Training - FitNet
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        METHOD=FitNet
        PARAM=1.5,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
        python train_student.py --seed 5 --gpus 6 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}
    done
done


# Student Training - RKD
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        METHOD=RKD
        PARAM=40.0,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
        python train_student.py --seed 5 --gpus 7 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}
    done
done