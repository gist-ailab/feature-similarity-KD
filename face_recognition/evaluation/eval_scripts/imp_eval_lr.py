import os
import subprocess
import numpy as np
import pickle
import argparse
from glob import glob


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    gpus=args.gpus
    save_dir = "./result/imp/eval2"
    os.makedirs(save_dir, exist_ok=True)

    stage = 2
    
    if stage == 1:
        ckpt_path = '/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/final_ablation/casia/case2'
        ckpt_list = glob(os.path.join(ckpt_path, '*/*/last_net.ckpt'))

        mode = 'ir' 
        for ckpt in ckpt_list:
            conditions = '-'.join(ckpt.split('/')[-3].split('-')[2:])
            margins = ckpt.split('/')[-2].split('-')[1]
            
            ckpt = os.path.dirname(ckpt)
            
            prefix = "%s_%s_tiny" %(conditions, margins)
            subprocess.call("python test_tinyface.py --gpus %s --mode %s --save_dir %s --prefix %s \
                                                    --backbone iresnet50 --pooling E --checkpoint_path %s --qualnet %s --aligned %s" %(
                                                    gpus, mode, save_dir, prefix, ckpt, False, False), shell=True)
    
    else:
        result_list = []
        for M in ['CosFace', 'AdaFace']:
            for S in ['fix', 'range']:
                result_list.append('%s-%s \n' %(M, S))
                
                for P in [0.0, 0.2]:
                    r_ix = '%.1f' %P
                    
                    for L in [0.2, 0.5, 1.0]:
                        result_path = os.path.join(save_dir, 'PHOTO{%.1f}-LR{%.1f}-SIZE{%s}_%s_tiny.pkl' %(P, L, S, M))
                        with open(result_path, 'rb') as f:
                            result = pickle.load(f)
                        r_ix += ' %.2f' %result['rank1']
                    
                    r_ix += ' \n'
                    result_list.append(r_ix)
        
        with open('imp.txt' ,'w') as f:
            f.writelines(result_list)        