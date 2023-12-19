import os
import pickle

if __name__=='__main__':
    save_dir = "./result/table3"
    ckpt_dir = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/"
    
    # Summary    
    seed = 5
    margin = 'CosFace'
    for aligned in ['False', 'True']:
        result_list = []
        ckpt_list = [['naive_ir', 'Naive'], ['fitnet_ir', 'FitNet'], ['at_ir', 'AT'],
                     ['rkd_ir', 'RKD'], ['askd_cbam', 'A-SKD'], ['qualnet_qualnet', 'QualNet-LM'], ['fskd_ir', 'Ours']]
        for ckpt_set in ckpt_list:
            ckpt_name, prefix = ckpt_set
            ckpt_name = ckpt_name + '_%s_%d_aligned{%s}.pkl' %(margin, seed, aligned)
            
            # IJB-B
            ckpt_path = os.path.join(save_dir, 'IJBB', ckpt_name)
            with open(ckpt_path, 'rb') as f:
                ijbb_result = pickle.load(f)
            
            
            # IJB-C 
            ckpt_path = os.path.join(save_dir, 'IJBC', ckpt_name)
            with open(ckpt_path, 'rb') as f:
                ijbc_result = pickle.load(f)
            
            # TinyFace
            ckpt_path = os.path.join(save_dir, 'tinyface', ckpt_name)
            with open(ckpt_path, 'rb') as f:
                tiny_result = pickle.load(f)
            
            output = '%s &    & %.2f & %.2f & %.2f \n' %(prefix, tiny_result['rank1'], tiny_result['rank10'], tiny_result['rank20'])
            # output = '%s &    & %.2f & %.2f & %.2f \n' %(prefix, ijbb_result['0.001'], ijbc_result['0.001'], tiny_result['rank20'])
            result_list.append(output)
        
        with open(os.path.join(save_dir, 'aligned{%s}.txt' %aligned), 'w') as f:
            f.writelines(result_list)
        
        