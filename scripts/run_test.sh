# Test Code (AgeDB-30)
for PARAM in 0.0 4.0 8.0 12.0 16.0 20.0
do
    BACKBONE=iresnet50
    POOLING=E
    python test_tinyface.py --gpus 1 --backbone $BACKBONE --pooling $POOLING --checkpoint "checkpoint/student/$BACKBONE-$POOLING-IR/resol1-random/F_SKD_CROSS-P{20.0,$PARAM}/last_net.ckpt"
done


# Test Code (AgeDB-30)
RESOLUTION=0
python test_agedb.py --seed 0 --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset --backbone iresnet50 --pooling E --qualnet True \
                     --down_size $RESOLUTION --mode ir --checkpoint_dir checkpoint/teacher/iresnet50-E-qualnet-AdaFace


# Test Code (AgeDB-30)
RESOLUTION=0
python test_agedb.py --seed 0 --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset --backbone iresnet50 --pooling E --qualnet True \
                     --down_size $RESOLUTION --mode ir --checkpoint_dir checkpoint/teacher/iresnet50-E-qualnet-ArcFace


# Test Code (AgeDB-30)
RESOLUTION=0
python test_agedb.py --seed 0 --gpus 5 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset --backbone iresnet50 --pooling E --qualnet True \
                     --down_size $RESOLUTION --mode ir --checkpoint_dir checkpoint/teacher/iresnet50-E-qualnet-CosFace