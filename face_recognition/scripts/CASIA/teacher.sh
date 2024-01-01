# CASIA
for MARGIN in CosFace ArcFace AdaFace
do
    for S in 64.0
    do
        SEED=5
        if [ "$MARGIN" = "CosFace" ]; then
            M_F=0.4
        elif [ "$MARGIN" = "ArcFace" ]; then
            M_F=0.5
        elif [ "$MARGIN" = "AdaFace" ]; then
            M_F=0.4
        fi
        DATASET=casia
        RESOLUTION=0
        BACKBONE=iresnet50
        POOLING=E
        python train_teacher.py --gpus 0 --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$MARGIN-m{$M_F}-s{$S} \
                                --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256 \
                                --scale $S --margin_float $M_F
    done
done