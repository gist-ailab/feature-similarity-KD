#!/usr/bin/env python
# encoding: utf-8
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import random
from collections import defaultdict
import pickle
from dataset.augmenter import Augmenter
from PIL import Image

def img_loader(augmenter, path, down_size):
    try:
        with open(path, 'rb') as f:
            high_img = cv2.imread(path)
            if len(high_img.shape) == 2:
                high_img = np.stack([high_img] * 3, 2)
            
            if (down_size == 112) or (down_size == 0):
                down_img = high_img.copy()

            elif down_size == 1:
                down_img = high_img.copy()
                down_img = Image.fromarray(down_img)
                down_img = augmenter.augment(down_img)
                down_img = np.array(down_img)

            else:
                down_img = high_img
                
            return high_img, down_img
        
    except IOError:
        print('Cannot load image ' + path)
        

class FaceDataset(data.Dataset):
    def __init__(self, root, data_type, file_list, down_size, transform=None, loader=img_loader, flip=True, photo_prob=0.2, lr_prob=0.2, size_type='none', teacher_folder='', cross_sampling=False, margin=0.0):
        self.root = root
        self.data_type = data_type
        self.transform = transform
        self.loader = loader
        self.down_size = down_size
        
        self.augmenter = Augmenter(photometric_augmentation_prob=photo_prob, low_res_augmentation_prob=lr_prob, size_type=size_type)

        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        
        self.label_dict = defaultdict(list)
        for ind, info in enumerate(img_label_list):
            image_path, label_name = info.split('  ')
            image_list.append(image_path)
            label_list.append(int(label_name))
            
            self.label_dict[int(label_name)].append(ind)

        self.margin = margin
        if cross_sampling:
            if self.margin < -2:
                self.cross_dict = len(image_list)
            else:
                with open(os.path.join(teacher_folder, 'cross_dict_%s.pkl' %self.data_type), 'rb') as f:
                    self.cross_dict = pickle.load(f)
        else:
            self.cross_dict = None

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

        self.flip = flip
        print("dataset size: ", len(self.image_list), '/', self.class_nums)


    def __getitem__(self, index):
        HR_img, LR_img, label = self.get_samples(index)
        if self.cross_dict is None:
            return HR_img, LR_img, label
        else:
            if self.margin < -1:
                pos_index = np.random.choice(list(range(self.cross_dict)), 1, replace=False)[0]
                HR_pos_img, LR_pos_img, _ = self.get_samples(int(pos_index))
                correct_index = torch.tensor(1)
            else:
                pos_list = self.cross_dict[index]['pos_m{%.1f}' %self.margin]
                if len(pos_list) == 0:
                    HR_pos_img, LR_pos_img = HR_img, LR_img
                    correct_index = torch.tensor(0)
                else:
                    pos_index = np.random.choice(pos_list, 1, replace=False)[0]
                    HR_pos_img, LR_pos_img, _ = self.get_samples(int(pos_index))
                    correct_index = torch.tensor(1)
            
            return HR_img, LR_img, HR_pos_img, LR_pos_img, correct_index, label

    # def update_candidate(self):
    #     # self.candidate = np.random.choice([8, 16, 24, 32, 48, 64, 112], 3, replace=False).tolist()
    #     self.candidate = np.random.choice([14, 21, 28, 42, 56, 84, 112], 3, replace=False).tolist()
    #     print('Resolution: ', self.candidate)

    def get_samples(self, index):        
        img_path = self.image_list[index]

        if 'casia' in self.data_type:
            ind = img_path.find('faces_webface_112x112')
            img_path = img_path[ind+28:]
        elif 'ms1mv2' in self.data_type:
            ind = img_path.find('faces_emore')
            img_path = img_path[ind+18:]
        elif 'vggface' in self.data_type:
            ind = img_path.find('faces_vgg_112x112')
            img_path = img_path[ind+24:]
        elif 'webface4m' in self.data_type:
            ind = img_path.find('webface4m')
            img_path = img_path[ind+16:]
        else:
            raise('Error!')

        label = self.label_list[index]
        HR_img, LR_img = self.loader(self.augmenter, os.path.join(self.root, img_path), self.down_size)

        # random flip with ratio of 0.5
        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            if flip == 1:
                HR_img = cv2.flip(HR_img, 1)
                LR_img = cv2.flip(LR_img, 1)

        if self.transform is not None:
            HR_img = self.transform(HR_img)
            LR_img = self.transform(LR_img)
            
        else:
            HR_img = torch.from_numpy(HR_img)
            LR_img = torch.from_numpy(LR_img)

        return HR_img, LR_img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = 'D:/data/webface_align_112'
    file_list = 'D:/data/webface_align_train.list'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = CASIAWebFace(root, file_list, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)