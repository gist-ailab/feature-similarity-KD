# Teacher Training
for MARGIN in ArcFace CosFace AdaFace
do
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed 5 --gpus 0,1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-ms1mv2/$BACKBONE-$POOLING-IR-$MARGIN/ \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset ms1mv2 --batch_size 512
done


# Naive Training
for MARGIN in ArcFace CosFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        RESOLUTION=1
        POOLING=E
        INTERPOLATION=random
        python train_teacher.py --seed 5 --gpus 2,3 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-ms1mv2/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION \
                                --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset ms1mv2 --batch_size 512
    done
done


# # Student Training
# for MARGIN in CosFace ArcFace AdaFace
# do
#     for BACKBONE in iresnet50 iresnet18
#     do
#         METHOD=F_SKD_CROSS
#         PARAM=20.0,4.0
#         RESOLUTION=1
#         INTERPOLATION=random
#         POOLING=E
#         TEACHER=checkpoint/teacher/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
#         python train_student.py --seed 5 --gpus 2 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
#                                 --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
#                                 --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}
#     done
# done