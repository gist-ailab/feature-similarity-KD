# Teacher Training
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5 4 3 2 1
    do
        RESOLUTION=0
        BACKBONE=iresnet50
        POOLING=E
        python train_qualnet_stage1.py --seed $SEED --gpus 2 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-casia/$BACKBONE-$POOLING-qualnet-$MARGIN/seed{$SEED} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE
    done
done




# Student Training -> pretrained=True
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet50
        METHOD=QualNet
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        PRETRAINED=True
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-qualnet-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_qualnet_stage2.py --seed $SEED --gpus 4 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING --equal False \
                                --pretrained_student $PRETRAINED --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-qualnet-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-pretrained{$PRETRAINED}/seed{$SEED} \
                                --margin_float 0.2
    done
done


# Student Training -> pretrained=False
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5 4 3 2 1
    do
        BACKBONE=iresnet50
        METHOD=QualNet
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        PRETRAINED=False
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-qualnet-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_qualnet_stage2.py --seed $SEED --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING --equal False \
                                --pretrained_student $PRETRAINED --teacher_path $TEACHER --save_dir checkpoint/student-casia/$BACKBONE-$POOLING-qualnet-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-pretrained{$PRETRAINED}/seed{$SEED}
    done
done

