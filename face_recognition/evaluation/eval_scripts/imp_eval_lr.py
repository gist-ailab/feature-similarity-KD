import os
import subprocess
import numpy as np
import pickle
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--gpus', type=str, default='7')
    args = parser.parse_args()

    gpus=args.gpus
    save_dir = "./result/imp/eval"
    os.makedirs(save_dir, exist_ok=True)

    ckpt_dir = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/"
    for margin in ['CosFace']:
        for f_m in ['0.4', '0.3', '0.35', '0.2']:
            ckpt_path = "student-test/student-casia/iresnet50-E-IR-%s/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}-FM{%s}/seed{5}" %(
                margin, f_m)
            ckpt_path = os.path.join(ckpt_dir, ckpt_path)

            # QMulSurvFace
            prefix = "%s_%s_qmul" %(margin, f_m)
            mode = 'ir' 
            subprocess.call("python test_qmul.py --gpus %s --mode %s --save_dir %s --prefix %s \
                                    --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s" %(
                gpus, mode, save_dir, prefix, ckpt_path, False), shell=True)
                
                        
            for aligned in ['False', 'True']:
                prefix = "%s_%s_aligned{%s}_tiny" %(margin, f_m, aligned)
                mode = 'ir' 

                # TinyFace
                subprocess.call("python test_tinyface.py --gpus %s --mode %s --save_dir %s --prefix %s \
                                                        --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned %s" %(
                                    gpus, mode, save_dir, prefix, ckpt_path, False, aligned), shell=True)