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



# Cross Sampling = True (DDP = True)
for SEED in 5 4 3 2 1
do
    for MARGIN in CosFace ArcFace AdaFace
    do
        BACKBONE=iresnet50
        METHOD=F_SKD_CROSS_BN
        PARAM=20.0,4.0
        CMARGIN=0.0
        RESOLUTION=1
        INTERPOLATION=random
        POOLING=E
        DATASET=casia
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=991 train_student_multi.py --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --down_size $RESOLUTION \
                                                                    --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                                                                    --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER \
                                                                    --save_dir checkpoint/student-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/$METHOD-P{$PARAM}-M{$CMARGIN}/seed{$SEED} \
                                                                    --batch_size 256 --dataset $DATASET --cross_margin $CMARGIN --cross_sampling True --hint_bn True
    done
done