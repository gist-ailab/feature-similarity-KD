MARGIN=AdaFace
python train_embed.py --mode ir --backbone iresnet50 --gpus 2 --data_dir /home/jovyan/SSDb/sung/dataset/face_dset --teacher_path checkpoint/teacher-casia/iresnet50-$MARGIN/last_net.ckpt