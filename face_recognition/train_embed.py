#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import torch.utils.data
from backbone.iresnet import iresnet18, iresnet50
from torch.nn import DataParallel
from dataset.train_dataset import FaceDataset

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import random
from collections import OrderedDict
import pickle

def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def selection(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # validation dataset
    trainset = FaceDataset(args.train_root, args.dataset, args.train_file_list, 0, transform=transform, photo_prob=0.0, lr_prob=0.0, size_type='none')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # define backbone and margin layer
    net = iresnet50(attention_type=args.mode, pooling=args.pooling)

    # Load Pretrained Teacher
    net_ckpt = torch.load(args.teacher_path, map_location='cpu')['net_state_dict']
    new_ckpt = OrderedDict()
    for key, value in net_ckpt.items():
        if 'conv_bridge' not in key:
            new_ckpt[key] = value
    
    net.load_state_dict(new_ckpt)
    
    for param in net.parameters():
        param.requires_grad = False


    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    # Run
    net.eval()        
    feat_list, label_list = [], []

    for data in tqdm(trainloader):            
        HR_img, label = data[0].to(device), data[2].to(device)          
        
        with torch.no_grad():
            HR_logits = net(HR_img)

        feat_list.append(HR_logits)
        label_list.append(label)

    label_list = torch.cat(label_list, dim=0).cpu().detach()
    feat_list = torch.cat(feat_list, dim=0).cpu().detach()
    feat_list = feat_list / torch.norm(feat_list, dim=1).unsqueeze(dim=1)

    result = []
    for ii in tqdm(range(feat_list.size(0))):
        gt_ii = label_list[ii]
        gt_index = torch.where(label_list == gt_ii)[0]

        similarity_ii = (feat_list[[ii]] @ (feat_list.T)[:, gt_index])[0]
        similarity_ii[similarity_ii > 0.99] = 0.0

        out_dict = {}
        for margin in [0.0]:
            index = torch.where(similarity_ii > margin)[0]
            out_dict['pos_m{%.1f}' %margin] = gt_index[index].tolist()
        
        result.append(out_dict)
    
    with open(os.path.join(os.path.dirname(args.teacher_path), 'cross_dict_%s.pkl' %args.dataset), 'wb') as f:
        pickle.dump(result, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--dataset', type=str, default='webface4m')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset/')

    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E')
    
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gpus', type=str, default='7', help='model prefix')
    parser.add_argument('--seed', type=int, default=5)

    parser.add_argument('--teacher_path', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/teacher-webface4m/iresnet50-E-IR-CosFace/seed{5}/last_net.ckpt')
    args = parser.parse_args()


    # PATH
    if args.dataset == 'casia':
        args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    elif args.dataset == 'webface4m':
        args.train_root = os.path.join(args.data_dir, 'webface4m_subset/image')
        args.train_file_list = os.path.join(args.data_dir, 'webface4m_subset/train.list')
    elif args.dataset == 'ms1mv2':
        args.train_root = os.path.join(args.data_dir, 'faces_emore/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_emore/train.list')
    elif args.dataset == 'vggface':
        args.train_root = os.path.join(args.data_dir, 'faces_vgg_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_vgg_112x112/train.list')
    else:
        raise('Select Proper Dataset')
    
    # Seed
    set_random_seed(args.seed)
    
    # Run    
    selection(args)