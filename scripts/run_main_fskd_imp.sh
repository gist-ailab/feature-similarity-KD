# Student Training
BACKBONE=iresnet18
BN=False
GPU=3

MARGIN=CosFace
METHOD=F_SKD_CROSS
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/last_net.ckpt
python train_student.py --seed 5 --gpus $GPU --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/test/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-hint-BN{$BN} \
                        --batch_size 256 --dataset $DATASET --student_bn $BN