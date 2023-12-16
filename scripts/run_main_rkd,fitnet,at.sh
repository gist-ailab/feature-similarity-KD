# Student Training - AT
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet50
        METHOD=AT
        PARAM=1000,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py --seed $SEED --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --margin_float 0.2
    done
done

# MobileNet
for SEED in 5
do
    for MARGIN in CosFace ArcFace AdaFace
    do
        BACKBONE=mobilenet
        METHOD=AT
        PARAM=1000,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        DATASET=casia
        python train_student.py --seed $SEED --gpus 1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --margin_float 0.2
    done
done




# Student Training - FitNet
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet50
        METHOD=FitNet
        PARAM=1.5,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py --seed $SEED --gpus 6 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --margin_float 0.2
    done
done


# Student Training - FitNet (MobileNet)
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=mobilenet
        METHOD=FitNet
        PARAM=1.5,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py --seed $SEED --gpus 2 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --margin_float 0.2
    done
done



# Student Training - RKD
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=mobilenet
        METHOD=RKD
        PARAM=40.0,0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py --seed $SEED --gpus 3 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                                --margin_float 0.2
    done
done