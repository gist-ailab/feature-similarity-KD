## Teacher Training
for SEED in 5
do
    MARGIN=CosFace
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed $SEED --gpus 0 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256
done


# Naive Training - resnet
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in iresnet18
    do
        for SEED in 5
        do
            DATASET=casia
            RESOLUTION=1
            POOLING=E
            INTERPOLATION=random
            python train_teacher.py --seed $SEED --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                                    --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 256 --margin_float 0.2
        done
    done
done


# Naive Training
for MARGIN in CosFace ArcFace AdaFace
do
    for BACKBONE in mobilenet
    do
        for SEED in 5
        do
            DATASET=casia
            RESOLUTION=1
            POOLING=E
            INTERPOLATION=random
            python train_teacher.py --seed $SEED --gpus 4 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                                    --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 256 --margin_float 0.2
        done
    done
done



# Cross Sampling = True (DDP = True, MODE=IR)
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        CMARGIN=0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=991 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                    --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                                                    --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                    --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                                                    --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True --margin_float 0.2
    done
done


# Cross Sampling = True (DDP = True, MODE=CBAM)
for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        CMARGIN=0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-CBAM-$MARGIN/seed{$SEED}/last_net.ckpt
        CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=911 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                    --backbone $BACKBONE --mode cbam --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                                                    --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                    --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-CBAM-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                                                    --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True --margin_float 0.2
    done
done



# Extras (iresnet50 -> iresnet18) # Interpolattion=Random
# for MARGIN in CosFace ArcFace AdaFace
# do
#     for SEED in 5
#     do
#         BACKBONE=iresnet18
#         METHOD=F_SKD_CROSS_BN
#         PARAM=20.0,4.0
#         CMARGIN=0.0
#         RESOLUTION=1
#         INTERPOLATION=random
#         POOLING=E
#         DATASET=casia
#         TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
#         CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=11 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
#                                                                     --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
#                                                                     --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
#                                                                     --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
#                                                                     --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True
#     done
# done

for MARGIN in CosFace ArcFace AdaFace
do
    for SEED in 5
    do
        BACKBONE=iresnet18
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        CMARGIN=0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        python train_student.py  --gpus 6 --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                       --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                                       --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER\
                                                       --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                                       --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True --margin_float 0.2
    done
done


# Extras (MobileNet)
for SEED in 5
do
    for MARGIN in CosFace ArcFace AdaFace
    do
        BACKBONE=mobilenet
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        CMARGIN=0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        TEACHER=checkpoint/teacher-casia/iresnet50-$POOLING-IR-$MARGIN/seed{$SEED}/last_net.ckpt
        CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=981 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                    --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                                                    --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                    --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                                                    --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True --margin_float 0.2
    done
done


