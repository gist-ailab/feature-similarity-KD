import os
import subprocess
import numpy as np
import pickle
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--gpus', type=str, default='1')
    args = parser.parse_args()
    
    stage = 2

    gpus=args.gpus
    save_dir = "./result/table4/agedb"
    os.makedirs(save_dir, exist_ok=True)

    ckpt_dir = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/"
    
    if stage == 1:
        for backbone in ['mobilenet']:
            for margin in ['CosFace']:
                ckpt_list = [
                                [os.path.join(ckpt_dir, 'naive-casia/%s-E-IR-%s/resol1-random' %(backbone, margin)), 'ir', 'naive-%s'%backbone],
                                [os.path.join(ckpt_dir, 'student-casia/%s-E-IR-%s/resol1-random/FitNet-P{1.5,0.0}' %(backbone, margin)), 'ir', 'fitnet-%s'%backbone],
                                [os.path.join(ckpt_dir, 'student-casia/%s-E-IR-%s/resol1-random/AT-P{1000,0.0}' %(backbone, margin)), 'ir', 'at-%s'%backbone],
                                [os.path.join(ckpt_dir, 'student-casia/%s-E-IR-%s/resol1-random/RKD-P{40.0,0.0}' %(backbone, margin)), 'ir', 'rkd-%s'%backbone],
                                [os.path.join(ckpt_dir, 'student-casia/%s-E-IR-%s/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}' %(backbone, margin)), 'ir', 'fskd-%s'%backbone]
                            ]
                
                for ckpt_set in ckpt_list:
                    seed = 5
                    ckpt_path, mode, name = os.path.join(ckpt_set[0], 'seed{%d}' %seed), ckpt_set[1], ckpt_set[2]

                    prefix = "%s_%s_%s_%d" %(name, mode, margin, seed)

                    if mode == 'ir':
                        settings = ['ir', 'False']
                    elif mode == 'qualnet':
                        settings = ['ir', 'True']
                    elif mode == 'cbam':
                        settings = ['cbam', 'False']
                    else:
                        raise('Error!')
                    
                    # AgeDB-30
                    subprocess.call("python test_agedb.py --down_size 1 --checkpoint_dir %s \
                                                        --save_dir %s --prefix %s --eval_dataset agedb30 --eval_cross_resolution False \
                                                        --mode %s --backbone %s --pooling E --batch_size 256 --gpus %s \
                                                        --qualnet %s --seed %d" %(ckpt_path, save_dir, prefix, settings[0], backbone, gpus,
                                                                                   settings[1], seed), shell=True)
                
    else:
        ckpt_list = [['naive-mobilenet_ir', 'Naive'], ['fitnet-mobilenet_ir', 'FitNet'], ['at-mobilenet_ir', 'AT'], ['rkd-mobilenet_ir', 'RKD'], ['fskd-mobilenet_ir', 'Ours']]
        seed = 5
        margin='CosFace'
        result_list = [] 
        for ckpt_set in ckpt_list:
            ckpt_name, prefix = ckpt_set[0], ckpt_set[1]
            ckpt_path = ckpt_name + '_%s_%d.pkl' %(margin, seed)
            with open(os.path.join(save_dir, ckpt_path), 'rb') as f:
                result = pickle.load(f)['agedb30']
            output = '%s & %.2f & %.2f & %.2f & %.2f & %.2f' %(prefix, result['112'], result['56'], result['28'], result['14'], (result['112'] + result['56'] + result['28'] + result['14'])/4)
            result_list.append(output)
        print(result_list)