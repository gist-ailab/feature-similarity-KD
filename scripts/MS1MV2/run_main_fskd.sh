## Teacher Training
for MARGIN in CosFace ArcFace AdaFace
do
    SEED=5

    # Teacher Train
    # Set MARGIN_F based on the MARGIN value
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.4
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi
    DATASET=ms1mv2
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --seed $SEED --gpus 0 --data_dir /data/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/seed{$SEED} \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 512 --margin_float $MARGIN_F


    # Naive Train
    DATASET=ms1mv2
    RESOLUTION=1
    POOLING=E
    INTERPOLATION=random
    python train_teacher.py --seed $SEED --gpus 0 --data_dir /data/sung/dataset/face_dset/ --save_dir checkpoint/naive-$DATASET/$BACKBONE-$POOLING-IR-$MARGIN/resol$RESOLUTION-$INTERPOLATION/seed{$SEED} \
                            --backbone $BACKBONE --down_size $RESOLUTION --pooling $POOLING --interpolation $INTERPOLATION --mode ir --margin_type $MARGIN --dataset $DATASET --batch_size 512 --margin_float 0.2
done
