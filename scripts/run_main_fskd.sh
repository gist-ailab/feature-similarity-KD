## Teacher Training
for SEED in 5 4 3 2 1
do
    MARGIN=CosFace
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed $SEED --gpus 0 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256
done


# Naive Training
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet50
    do
        for SEED in 5 4 3 2 1
        do
            DATASET=casia
            RESOLUTION=1
            POOLING=E
            INTERPOLATION=random
            python train_teacher.py --seed $SEED --gpus 3 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                                    --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 256
        done
    done
done






################################### Student Training ##########################
# Cross Sampling = False
for SEED in 5
do
    MARGIN=CosFace
    BACKBONE=iresnet50
    METHOD=F_SKD_BN
    PARAM=20.0,4.0
    RESOLUTION=1
    INTERPOLATION=random
    POOLING=E
    DATASET=casia
    TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
    python train_student.py --seed $SEED --gpus 4 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                            --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                            --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}/seed{$SEED} \
                            --batch_size 256 --dataset $DATASET --cross_sampling False
done



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




# Cross Sampling = True (DDP = True)
for CMARGIN in -1.0 0.8
do
    for SEED in 5
    do
        MARGIN=CosFace
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student_multi.py --seed $SEED --gpus 0,1 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --port 901
    done
done


for CMARGIN in 0.0 0.6
do
    for SEED in 5
    do
        MARGIN=CosFace
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student_multi.py --seed $SEED --gpus 4,5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --port 886
    done
done



for CMARGIN in 0.2 0.4
do
    for SEED in 5
    do
        MARGIN=CosFace
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-$DATASET/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student_multi.py --seed $SEED --gpus 6,7 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --port 991
    done
done