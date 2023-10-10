# Test Code (AgeDB-30)
for MARGIN in ArcFace CosFace AdaFace
do
    BACKBONE=iresnet18
    POOLING=E
    python test_tinyface.py --gpus 1 --backbone $BACKBONE --pooling $POOLING --checkpoint "checkpoint/student/$BACKBONE-$POOLING-IR-$MARGIN/resol1-random/F_SKD_CROSS-P{20.0,4.0}/last_net.ckpt"
    # python test_tinyface.py --gpus 1 --qualnet True --backbone $BACKBONE --pooling $POOLING --checkpoint "checkpoint/student/$BACKBONE-$POOLING-qualnet-$MARGIN/resol1-random/QualNet-pretrained{True}/last_net.ckpt"
    # python test_tinyface.py --gpus 3 --backbone $BACKBONE --mode cbam --pooling $POOLING --checkpoint "checkpoint/student/$BACKBONE-$POOLING-CBAM-$MARGIN/resol1-random/A_SKD-P{80.0,0.0}/last_net.ckpt"
    # python test_tinyface.py --gpus 0 --backbone $BACKBONE --pooling $POOLING --checkpoint "checkpoint/naive/$BACKBONE-$POOLING-IR-$MARGIN/resol1-random/last_net.ckpt"
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