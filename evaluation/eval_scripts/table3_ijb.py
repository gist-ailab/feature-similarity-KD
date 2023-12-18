import os
import subprocess
import numpy as np
import pickle
import argparse

# {'0.01': 79.37, '0.001': 59.28}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='IJBB')
    args = parser.parse_args()

    gpus = args.gpus
    save_dir = "./result/table3/%s" %args.dataset
    os.makedirs(save_dir, exist_ok=True)

    ckpt_dir = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/"

    margin = 'CosFace'
    ckpt_list = [
                    [os.path.join(ckpt_dir, 'naive-casia/iresnet50-E-IR-%s/resol1-random' %margin), 'ir', 'naive'],
                    [os.path.join(ckpt_dir, 'student-casia/iresnet50-E-IR-%s/resol1-random/FitNet-P{1.5,0.0}' %margin), 'ir', 'fitnet'],
                    [os.path.join(ckpt_dir, 'student-casia/iresnet50-E-IR-%s/resol1-random/AT-P{1000,0.0}' %margin), 'ir', 'at'],
                    [os.path.join(ckpt_dir, 'student-casia/iresnet50-E-IR-%s/resol1-random/RKD-P{40.0,0.0}' %margin), 'ir', 'rkd'],
                    [os.path.join(ckpt_dir, "student-casia/iresnet50-E-CBAM-%s/resol1-random/A_SKD-P{80.0,0.0}" %margin), 'cbam', 'askd'],
                    [os.path.join(ckpt_dir, 'student-casia/iresnet50-E-qualnet-%s/resol1-random/QualNet-pretrained{True}' %margin), 'qualnet', 'qualnet'],
                    [os.path.join(ckpt_dir, 'student-casia/iresnet50-E-IR-%s/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}' %margin), 'ir', 'fskd'],
                ]
    
    for ckpt_set in ckpt_list:
        for seed in [5]:
            ckpt_path, mode, name = os.path.join(ckpt_set[0], 'seed{%d}' %seed), ckpt_set[1], ckpt_set[2]

            if mode == 'ir':
                settings = ['ir', 'False']
            elif mode == 'qualnet':
                settings = ['ir', 'True']
            elif mode == 'cbam':
                settings = ['cbam', 'False']
            else:
                raise('Error!')

            if args.dataset == 'IJBB':
                # IJB-B
                prefix = "%s_%s_%s_%d_aligned{False}" %(name, mode, margin, seed)
                subprocess.call("python test_ijb.py --data_name IJBB --gpus %s --save_dir %s --prefix %s --mode %s --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned False" %(
                    gpus, save_dir, prefix, settings[0], ckpt_path, settings[1]
                ), shell=True)
            
                # IJB-B
                prefix = "%s_%s_%s_%d_aligned{True}" %(name, mode, margin, seed)
                subprocess.call("python test_ijb.py --data_name IJBB --gpus %s --save_dir %s --prefix %s --mode %s --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned True" %(
                    gpus, save_dir, prefix, settings[0], ckpt_path, settings[1]
                ), shell=True)
                
                
            elif args.dataset == 'IJBC':
                # IJB-C
                prefix = "%s_%s_%s_%d_aligned{False}" %(name, mode, margin, seed)
                subprocess.call("python test_ijb.py --data_name IJBC --gpus %s --save_dir %s --prefix %s --mode %s --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned False" %(
                    gpus, save_dir, prefix, settings[0], ckpt_path, settings[1]
                ), shell=True)

                # IJB-C
                prefix = "%s_%s_%s_%d_aligned{True}" %(name, mode, margin, seed)
                subprocess.call("python test_ijb.py --data_name IJBC --gpus %s --save_dir %s --prefix %s --mode %s --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned True" %(
                    gpus, save_dir, prefix, settings[0], ckpt_path, settings[1]
                ), shell=True)

            else:
                raise('Error!')
        
        