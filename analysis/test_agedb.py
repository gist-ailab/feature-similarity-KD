#!/usr/bin/env python
# encoding: utf-8
import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import sys
sys.path.append('../')

import torch.utils.data
from backbone.iresnet import iresnet18, iresnet50
from torch.nn import DataParallel
from dataset.agedb import AgeDB30
from dataset.cfp import CFP_FP
from dataset.lfw import LFW

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
from evaluation.eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import random


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


def cosine_loss(l , h):
    l = l.view(l.size(0), -1)
    h = h.view(h.size(0), -1)
    return torch.mean(1.0 - F.cosine_similarity(l, h))


def inference(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)


    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # define backbone and margin layer
    if args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)
    elif args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)

    # Load Pretrained Teacher
    net_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'last_net.ckpt'), map_location='cpu')['net_state_dict']
    net.load_state_dict(net_ckpt)
    
    for param in net.parameters():
        param.requires_grad = False
    
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    # test dataset
    net.eval()
    print('Evaluation on LFW, AgeDB-30. CFP')
    os.makedirs(os.path.join(args.checkpoint_dir, 'result'), exist_ok=True)
    
    
    from metric.distill_loss import normalize, cross_kd
    util = cross_kd()
    
    if args.down_size == 1:
        eval_list = [112, 56, 28, 14]
    elif args.down_size == 0:
        eval_list = [112]
    else: 
        eval_list = [args.down_size]
        
    for down_size in eval_list:
        agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform)
        agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        featureLs = None
        featureRs = None
        for data in agedbloader:
            L_img, R_img = data[0].to(device), data[2].to(device)
            B = L_img.size(0)
            with torch.no_grad():
                L_out = net(L_img, extract_feature=True)
                R_out = net(R_img, extract_feature=True)
                
                if args.qualnet:
                    L_logit, R_logit = L_out[0], R_out[0]
                    L_feature, R_feature = L_out[2][-1], R_out[2][-1]                    
                else:
                    L_logit, R_logit = L_out[0], R_out[0]
                    L_feature, R_feature = L_out[1][-1], R_out[1][-1]
            
            affinity = util.relation(L_feature, R_feature)
            B, N, _ = affinity.size()
            affinity = affinity.view(B, N, int(N**0.5), int(N**0.5))
            print(affinity.size())
            exit()

            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)

            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/student/iresnet50-E-IR-ArcFace/resol1-random/F_SKD_CROSS-P{20.0,4.0}', help='model save dir')
    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gpus', type=str, default='3', help='model prefix')
    parser.add_argument('--qualnet', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()


    # Path
    args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
    args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    args.lfw_test_root = os.path.join(args.data_dir, 'evaluation/lfw')
    args.lfw_file_list = os.path.join(args.data_dir, 'evaluation/lfw.txt')
    args.agedb_test_root = os.path.join(args.data_dir, 'evaluation/agedb_30')
    args.agedb_file_list = os.path.join(args.data_dir, 'evaluation/agedb_30.txt')
    args.cfpfp_test_root = os.path.join(args.data_dir, 'evaluation/cfp_fp')
    args.cfpfp_file_list = os.path.join(args.data_dir, 'evaluation/cfp_fp.txt')


    # Seed
    set_random_seed(args.seed)
    
    # Run    
    inference(args)