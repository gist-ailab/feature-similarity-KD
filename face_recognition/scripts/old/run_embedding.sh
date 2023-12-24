for SEED in 5 4 3 2 1
do
    for MARGIN in CosFace ArcFace AdaFace
    do
        python train_embed.py --mode cbam --backbone iresnet50 --gpus 0 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset --teacher_path checkpoint/teacher-casia/iresnet50-E-CBAM-$MARGIN/seed{$SEED}/last_net.ckpt
    done
done