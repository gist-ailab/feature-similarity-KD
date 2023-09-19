#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

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
    
    if args.down_size == 1:
        eval_list = [112, 56, 28, 14]
    elif args.down_size == 0:
        eval_list = [112]
    else: 
        eval_list = [args.down_size]
        
    average_age = 0.
    average_cfp = 0.
    average_lfw = 0.
    for down_size in eval_list:
        agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform)
        cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, down_size, transform=transform)
        lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, down_size, transform=transform)
        
        agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        # test model on AgeDB30
        getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader, qualnet=args.qualnet)
        age_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_agedb30_result.mat'))
        print('Evaluation Result on AgeDB-30 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))

        # test model on CFP-FP
        getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_cfpfp_result.mat'), net, device, cfpfpdataset, cfpfploader, qualnet=args.qualnet)
        cfp_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_cfpfp_result.mat'))
        print('Evaluation Result on CFP-ACC %dX - %.2f' %(down_size, np.mean(cfp_accs) * 100))
        
        # test model on LFW
        getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_lfw_result.mat'), net, device, lfwdataset, lfwloader, qualnet=args.qualnet)
        lfw_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_lfw_result.mat'))
        print('Evaluation Result on LFW-ACC %dX - %.2f' %(down_size, np.mean(lfw_accs) * 100))

        # Average
        average_age += np.mean(age_accs) * 100
        average_cfp += np.mean(cfp_accs) * 100
        average_lfw += np.mean(lfw_accs) * 100
        
        
    if len(eval_list) > 1:
        print('average - age_accs : %.2f' %(average_age / len(eval_list)))
        print('average - cfp_accs : %.2f' %(average_cfp / len(eval_list)))
        print('average - lfw_accs : %.2f' %(average_lfw / len(eval_list)))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/teacher/iresnet50-ir', help='model save dir')
    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--backbone', type=str, default='iresnet18')
    parser.add_argument('--pooling', type=str, default='A')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gpus', type=str, default='5', help='model prefix')
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