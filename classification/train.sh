# Teacher
CUDA_VISIBLE_DEVICES=1 python train_teacher.py --lr 0.1 --save-dir teacher_epoch200_lr{0.1}
CUDA_VISIBLE_DEVICES=2 python train_teacher.py --lr 0.05 --save-dir teacher_epoch200_lr{0.05}
CUDA_VISIBLE_DEVICES=3 python train_teacher.py --lr 0.01 --save-dir teacher_epoch200_lr{0.01}

# Naive
CUDA_VISIBLE_DEVICES=3 python train_teacher.py --resolution 8 --lr 0.1 --arch resnet20 --save-dir checkpoint/naive/epoch200_lr{0.1}_resol8
CUDA_VISIBLE_DEVICES=3 python train_teacher.py --resolution 1 --lr 0.1 --arch resnet20 --save-dir checkpoint/naive/epoch200_lr{0.1}_resol1

# Student
CUDA_VISIBLE_DEVICES=1 python train_student.py --resolution 1 --lr 0.1 --save-dir checkpoint/student/epoch200_lr{0.1}_resol1
CUDA_VISIBLE_DEVICES=2 python train_student.py --resolution 8 --lr 0.1 --save-dir checkpoint/student/epoch200_lr{0.1}_resol8
