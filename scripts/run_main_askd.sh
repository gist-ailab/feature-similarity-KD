# Teacher Training
for MARGIN in CosFace ArcFace AdaFace
do 
    for SEED in 5 4 3 2 1
    do
        RESOLUTION=0
        BACKBONE=iresnet50
        POOLING=E
        python train_teacher.py --seed $SEED --gpus 4 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-casia/$BACKBONE-$POOLING-CBAM-$MARGIN/seed{$SEED} \
                                --down_size $RESOLUTION --mode cbam --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE
    done
done


# Student Training
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5 4 3 2 1
    do
        METHOD=A_SKD
        BACKBONE=iresnet50
        PARAM=80.0,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-CBAM-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py --seed $SEED --gpus 1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode cbam --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-CBAM-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --cross_sampling False --hint_bn False
    done
done