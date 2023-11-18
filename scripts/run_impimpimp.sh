# DDP -> 2GPUs w/o Sync BN
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student_multi.py --seed $SEED --gpus 0,1 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_2gpu \
                        --batch_size 256 --dataset $DATASET --cross_sampling False --port 111 --sync False

# DDP -> 2GPUs + Sync BN
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student_multi.py --seed $SEED --gpus 0,1 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_2gpu_sync \
                        --batch_size 256 --dataset $DATASET --cross_sampling False --port 112 --sync True



# DP -> 2GPUs
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student.py --seed $SEED --gpus 0,1 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_2gpu_DP \
                        --batch_size 256 --dataset $DATASET --cross_sampling False


# 1gpu
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student.py --seed $SEED --gpus 0 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_1gpu \
                        --batch_size 256 --dataset $DATASET --cross_sampling False




############################################ Cross Sampling #######################################
for CMARGIN in 0.0
do
    MARGIN=CosFace
    BACKBONE=iresnet50
    METHOD=F_SKD_BN
    PARAM=20.0,4.0
    RESOLUTION=1
    INTERPOLATION=random
    POOLING=E
    DATASET=casia
    SEED=5
    TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
    python train_student_multi.py --seed $SEED --gpus 0,1 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                            --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                            --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_1gpu_cmargin{$CMARGIN} \
                            --batch_size 256 --dataset $DATASET --cross_sampling True --cross_margin $CMARGIN --port 11
done


MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
CMARGIN=0.2
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student.py --seed $SEED --gpus 0 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_1gpu_cmargin{$CMARGIN} \
                        --batch_size 256 --dataset $DATASET --cross_sampling True --cross_margin $CMARGIN


MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
CMARGIN=0.4
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student.py --seed $SEED --gpus 0 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp/student_casia_1gpu_cmargin{$CMARGIN} \
                        --batch_size 256 --dataset $DATASET --cross_sampling True --cross_margin $CMARGIN

















############################################ Pyramid Aug ###########################################
## Naive Training
MARGIN=CosFace
SEED=5
DATASET=casia
RESOLUTION=1
BACKBONE=iresnet50
POOLING=E
python train_teacher.py --seed $SEED --gpus 0 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --save_dir checkpoint/imp3/naive-caisa \
                        --down_size $RESOLUTION --mode ir --margin_type $MARGIN --pooling $POOLING --backbone $BACKBONE --dataset $DATASET --batch_size 256


## Student
MARGIN=CosFace
BACKBONE=iresnet50
METHOD=F_SKD_BN
PARAM=20.0,4.0
RESOLUTION=1
INTERPOLATION=random
POOLING=E
DATASET=casia
SEED=5
TEACHER=checkpoint/teacher-casia/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt
python train_student.py --seed $SEED --gpus 1 --data_dir /home/work/Workspace/sung/dataset/face_dset/ --down_size $RESOLUTION \
                        --backbone $BACKBONE --mode ir --interpolation $INTERPOLATION --margin_type $MARGIN --pooling $POOLING \
                        --distill_type $METHOD --distill_param $PARAM --teacher_path $TEACHER --save_dir checkpoint/imp3/student_casia_1gpu \
                        --batch_size 256 --dataset $DATASET --cross_sampling False
