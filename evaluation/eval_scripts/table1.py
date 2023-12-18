import os
import subprocess
import numpy as np
import pickle

if __name__=='__main__':
    stage = 2
    gpus='0'
    save_dir = "./result/table1/"

    if stage == 1:
        ckpt_dir = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/teacher-casia"
        os.makedirs(save_dir, exist_ok=True)

        for margin in ['CosFace', 'ArcFace', 'AdaFace']:
            for seed in [1,2,3,4,5]:
                ckpt_path = os.path.join(ckpt_dir, "iresnet50-E-IR-%s/seed{%d}" %(margin, seed))
                prefix = "%s_%d" %(margin, seed)
                down_size = 0
                subprocess.call("python test_agedb.py --down_size %d --checkpoint_dir %s \
                                                    --save_dir %s --prefix %s --eval_dataset all --eval_cross_resolution False \
                                                    --mode ir --backbone iresnet50 --pooling E --batch_size 256 --gpus %s \
                                                    --qualnet False --seed %d" %(down_size, ckpt_path, save_dir, prefix, gpus, seed), shell=True)
                
    else:
        print('------------ AgeDB30 -------------')
        for margin in ['CosFace', 'ArcFace', 'AdaFace']:
            result_list = []
            for seed in [5]:
                with open(os.path.join(save_dir, '%s_%d.pkl' %(margin, seed)), 'rb') as f:
                    result = pickle.load(f)
                
                result_list.append(result['agedb30']['112'])
            
            result_list = np.array(result_list)
            mu, std = np.mean(result_list), np.std(result_list)
            print('margin: %s, mean : %.2f, std : %.2f' %(margin, mu, std))


        print('------------ LFW -------------')
        for margin in ['CosFace', 'ArcFace', 'AdaFace']:
            result_list = []
            for seed in [5]:
                with open(os.path.join(save_dir, '%s_%d.pkl' %(margin, seed)), 'rb') as f:
                    result = pickle.load(f)
                
                result_list.append(result['lfw']['112'])
            
            result_list = np.array(result_list)
            mu, std = np.mean(result_list), np.std(result_list)
            print('margin: %s, mean : %.2f, std : %.2f' %(margin, mu, std))


        print('------------ CFP-FP -------------')
        for margin in ['CosFace', 'ArcFace', 'AdaFace']:
            result_list = []
            for seed in [5]:
                with open(os.path.join(save_dir, '%s_%d.pkl' %(margin, seed)), 'rb') as f:
                    result = pickle.load(f)
                
                result_list.append(result['cfp']['112'])
            
            result_list = np.array(result_list)
            mu, std = np.mean(result_list), np.std(result_list)
            print('margin: %s, mean : %.2f, std : %.2f' %(margin, mu, std))