#!/usr/bin/env python
# encoding: utf-8
from PIL import Image
import pickle
import mxnet as mx
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from mxdataset import MXDataset
import numpy as np
import cv2
'''
For train dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images
'''

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'image')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = os.path.join(save_path, str(label).zfill(6))
        
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)


def tensor_to_numpy(tensor):
    # -1 to 1 tensor to 0-255
    arr = tensor.numpy().transpose(1,2,0)
    return ((arr * 0.5 + 0.5) * 255).astype(np.uint8)

def load_mx_rec_webface4m(rec_path):
    save_path = os.path.join(rec_path, 'image')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset = MXDataset(root_dir=rec_path)
    dataloader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=False)
    for batch in tqdm(dataloader):
        imgs, tgts = batch
        count = {}
        for image, tgt in zip(imgs, tgts):
            label = str(tgt.item())
            image_uint8 = tensor_to_numpy(image)
            if label not in count:
                count[label] = []
            count[label].append(label)
            image_save_path = os.path.join(save_path, str(tgt.item()), f'{len(count[label])}.jpg')
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            cv2.imwrite(image_save_path, image_uint8)


def load_image_from_bin(bin_path, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file = open(os.path.join(save_dir, '../', '%s.txt' %name), 'w')
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, str(idx+1).zfill(5)+'.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx//2] == True else -1
            file.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')


def generate_dataset_list(dataset_path,dataset_list):
     label_list = os.listdir(dataset_path)
     f=open(dataset_list,'w')
     k=0

     for i in tqdm(label_list):
         label_path=os.path.join(dataset_path,i)
         if os.listdir(label_path):
            image_list=os.listdir(label_path)
            for j in image_list:
                image_path=os.path.join(label_path, j)
                f.write(image_path+'  '+str(k)+'\n')
         k=k+1
     f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/SSDb/sung/dataset/face_dset')
    parser.add_argument('--data_name', type=str, default='webface4m_subset')
    parser.add_argument('--data_type', type=str, default='train', help='train or evaluation')
    args = parser.parse_args()
    
    data_type = args.data_type
    data_dir = args.data_dir
    
    if data_type == 'evaluation':
        bin_path = os.path.join(data_dir, args.data_name, 'agedb_30.bin')
        save_dir = os.path.join(data_dir, 'evaluation', 'agedb_30')
        name = 'agedb_30'
        load_image_from_bin(bin_path, save_dir, name)

        bin_path = os.path.join(data_dir, args.data_name, 'lfw.bin')
        save_dir = os.path.join(data_dir, 'evaluation', 'lfw')
        name = 'lfw'
        load_image_from_bin(bin_path, save_dir, name)
        
        bin_path = os.path.join(data_dir, args.data_name, 'cfp_fp.bin')
        save_dir = os.path.join(data_dir, 'evaluation', 'cfp_fp')
        name = 'cfp_fp'
        load_image_from_bin(bin_path, save_dir, name)


    elif data_type == 'train':
        rec_path = os.path.join(data_dir, args.data_name)
        # if 'webface4m' in args.data_name:
        #     load_mx_rec_webface4m(rec_path)
        # else:
        #     load_mx_rec(rec_path)
        
        dataset = os.path.join(data_dir, args.data_name, 'image')
        list = os.path.join(data_dir, args.data_name, 'train.list')
        generate_dataset_list(dataset, list)
    else:
        raise('Error!')