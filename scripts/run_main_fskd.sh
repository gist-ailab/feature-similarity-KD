## Teacher Training
for MARGIN in CosFace ArcFace AdaFace
do
    SEED=5
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed $SEED --gpus 3,4 --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256
done


for MARGIN in CosFace
do
    SEED=5
    DATASET=vggface
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed $SEED --gpus 3,4,5,6 --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512
done





# Naive Training
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet50 iresnet18
    do
        SEED=5
        DATASET=casia
        RESOLUTION=1
        POOLING=E
        INTERPOLATION=random
        python train_teacher.py --seed $SEED --gpus 5,6 --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                                --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 256
    done
done


for MARGIN in CosFace
do
    for BACKBONE in iresnet50
    do
        SEED=5
        DATASET=vggface
        RESOLUTION=1
        POOLING=E
        INTERPOLATION=random
        python train_teacher.py --seed $SEED --gpus 3,4,5,6 --data_dir /SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                                --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 512
    done
done



################################### Student Training ##########################
# Cross Sampling = False
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
python train_student.py --seed $SEED --gpus 3,4 --data_dir /SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                        --batch_size 256 --dataset $DATASET --cross_sampling False


MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=vggface
SEED=5
TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
python train_student.py --seed $SEED --gpus 3,4,5,6 --data_dir /SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                        --batch_size 512 --dataset $DATASET --cross_sampling False



# Cross Sampling = True
for CMARGIN in 0.0 0.2 0.4
do
    MARGIN=CosFace
    BACKBONE=iresnet50
    METHOD=F_SKD_CROSS_BN
    PARAM=20.0,4.0
    RESOLUTION=1
    INTERPOLATION=random
    POOLING=E
    DATASET=casia
    SEED=5
    TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
    python train_student.py --seed $SEED --gpus 3,4 --data_dir /SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                            --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                            --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                            --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True
done
