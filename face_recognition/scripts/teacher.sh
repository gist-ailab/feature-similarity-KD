# CASIA
for MARGIN in AdaFace CosFace ArcFace
do
    SEED=1
    if [ "$MARGIN" = "CosFace" ]; then
        MARGIN_F=0.4
    elif [ "$MARGIN" = "ArcFace" ]; then
        MARGIN_F=0.5
    elif [ "$MARGIN" = "AdaFace" ]; then
        MARGIN_F=0.4
    fi
    DATASET=casia
    RESOLUTION=0
    BACKBONE=iresnet50
    POOLING=E
    python train_teacher.py --gpus 3 --seed $SEED --data_dir /home/jovyan/SSDb/sung/dataset/face_dset/ --save_dir checkpoint/teacher-$DATASET/$BACKBONE-$MARGIN \
                            --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256 --margin_float $MARGIN_F
done

