python train_embed.py --mode ir --backbone iresnet50 --gpus 0 --teacher_path checkpoint/teacher-casia/iresnet50-E-IR-CosFace/last_net.ckpt
python train_embed.py --mode ir --backbone iresnet50 --gpus 1 --teacher_path checkpoint/teacher-casia/iresnet50-E-IR-ArcFace/last_net.ckpt
