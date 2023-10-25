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


def img_loader(path, down_size, interpolation_option=None):
    try:
        with open(path, 'rb') as f:
            high_img = cv2.imread(path)
            if len(high_img.shape) == 2:
                high_img = np.stack([high_img] * 3, 2)
            
            if down_size != 112:
                # Down-Sampling
                if interpolation_option == 'random':
                    interpolation = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4], 1)[0]
                elif interpolation_option == 'fix':
                    interpolation = cv2.INTER_LINEAR
                else:
                    raise('Error')
                down_img = cv2.resize(high_img, dsize=(down_size, down_size), interpolation=interpolation)
                
                # Up-Sampling
                if interpolation_option == 'random':
                    interpolation = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4], 1)[0]
                elif interpolation_option == 'fix':
                    interpolation = cv2.INTER_LINEAR
                else:
                    raise('Error')
                down_img = cv2.resize(down_img, dsize=(112, 112), interpolation=interpolation)
                
            else:
                down_img = high_img
                
            return high_img, down_img
        
    except IOError:
        print('Cannot load image ' + path)
        

class FaceDataset(data.Dataset):
    def __init__(self, root, data_type, file_list, down_size, transform=None, loader=img_loader, flip=True, equal=True, interpolation_option=None, cross_sampling=False):
        self.root = root
        self.data_type = data_type
        self.transform = transform
        self.loader = loader
        self.down_size = down_size
        self.equal = equal
        
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

        self.cross_sampling = cross_sampling
        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

        self.interpolation_option = interpolation_option
        
        self.flip = flip
        print("dataset size: ", len(self.image_list), '/', self.class_nums)


    def __getitem__(self, index):
        HR_img, LR_img, label = self.get_samples(index)
        if self.cross_sampling:
            new_index = np.random.choice(self.label_dict[label], 1)[0]
            HR_cross_img, LR_cross_img, _ = self.get_samples(new_index)
            return HR_img, LR_img, HR_cross_img, LR_cross_img, label
        else:
            return HR_img, LR_img, label


    def get_samples(self, index):
        if self.down_size == 1:
            if self.equal:
                down_size = random.sample([14, 28, 56, 112], k=1)[0]
            else:
                choice = np.random.choice(['corrupt', 'none'], 1)[0]
                if choice == 'corrupt':
                    down_size = random.sample([14, 28, 56], k=1)[0]
                else:
                    down_size = 112

        elif self.down_size == 0:
            down_size = 112
        else:
            down_size = self.down_size
        
        img_path = self.image_list[index]

        if self.data_type == 'casia':
            ind = img_path.find('faces_webface_112x112')
            img_path = img_path[ind+28:]
        elif self.data_type == 'ms1mv2':
            ind = img_path.find('faces_emore')
            img_path = img_path[ind+18:]
        else:
            raise('Error!')

        label = self.label_list[index]
        HR_img, LR_img = self.loader(os.path.join(self.root, img_path), down_size, self.interpolation_option)

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