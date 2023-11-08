import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

def load_file(file_path):
    with open(file_path) as f:
        img_label_list = f.read().splitlines()
    
    data_dict = defaultdict(list)
    meta_dict = {}
    for info in tqdm(img_label_list):
        image_path, label_name = info.split('  ')
        image_path = '/'.join(image_path.split('/')[-2:])
        data_dict[str(label_name)].append(image_path)
        

    # Filter
    choice = []
    for id, paths in data_dict.items():
        meta_dict[id] = len(paths)
        
        if len(paths) > 20:
            choice.append(id)
            
    print('%d/%d' %(len(choice), len(data_dict.keys())))
    return data_dict, choice


if __name__=='__main__':
    root = '/home/jovyan/SSDb/sung/dataset/face_dset/faces_webface_112x112'
    file_path = os.path.join(root, 'train.list')
    data_dict, choice = load_file(file_path)
        
    # Split
    train_out, val_out = [], []
    for category_id, label_ix in enumerate(choice):
        image_paths = data_dict[label_ix]
        
        train_list = np.random.choice(image_paths, int(len(image_paths) * 0.7), replace=False)
        val_list = np.random.choice(list(set(image_paths) - set(train_list)), int(len(image_paths) * 0.3), replace=False)
        
        for path in train_list:
            train_out.append('%s  %s\n' %(os.path.join(root, 'image', path), category_id))
        for path in val_list: 
            val_out.append('%s  %s\n' %(os.path.join(root, 'image', path), category_id))
            
    
    # Save
    with open(os.path.join(root, 'train_mini.list'), 'w') as f:
        f.writelines(train_out)
    with open(os.path.join(root, 'val_mini.list'), 'w') as f:
        f.writelines(val_out)