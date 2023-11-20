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
for MARGIN in ArcFace CosFace AdaFace
do
    for BACKBONE in iresnet50
    do
        METHOD=QualNet
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        PRETRAINED=True
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-qualnet-$MARGIN/last_net.ckpt
        python train_qualnet_stage2.py --seed 5 --gpus 3 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING --equal False \
                                --pretrained_student $PRETRAINED --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-qualnet-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-pretrained{$PRETRAINED}
    done
done


# Student Training -> pretrained=False
for MARGIN in ArcFace CosFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        METHOD=QualNet
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        PRETRAINED=False
        TEACHER=checkpoint/teacher/iresnet50-$POOLING-qualnet-$MARGIN/last_net.ckpt
        python train_qualnet_stage2.py --seed 5 --gpus 4 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING --equal False \
                                --pretrained_student $PRETRAINED --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-qualnet-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-pretrained{$PRETRAINED}
    done
done