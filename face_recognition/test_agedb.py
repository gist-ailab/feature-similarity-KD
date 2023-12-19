#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import pickle
import torch.utils.data
from backbone.iresnet import iresnet18, iresnet50
from backbone.mobilenet import get_mbf_large
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
from collections import OrderedDict


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
    elif args.backbone == 'mobilenet':
        net = get_mbf_large(fp16=False, num_features=args.feature_dim)

    # Load Pretrained Teacher
    net_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'last_net.ckpt'), map_location='cpu')['net_state_dict']
    new_ckpt = OrderedDict()
    for key, value in net_ckpt.items():
        if ('conv_bridge' not in key) and ('hint' not in key):
            new_ckpt[key] = value
    net.load_state_dict(new_ckpt, strict=False)

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
        
    cross_resolution = args.eval_cross_resolution
    average_age = 0.
    average_cfp = 0.
    average_lfw = 0.

    if cross_resolution:
        print('Cross Resolution Evaluation')
    else:
        print('Single Resolution Evaluation')

    result = {'agedb30': {}, 'cfp': {}, 'lfw': {}}
    for down_size in eval_list:
        agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
        cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
        lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
        
        agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        # test model on AgeDB30
        if (args.eval_dataset == 'agedb30') or (args.eval_dataset == 'all'):
            getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader, qualnet=args.qualnet)
            age_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_agedb30_result.mat'))
            print('Evaluation Result on AgeDB-30 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))
            result['agedb30'][str(down_size)] = np.mean(age_accs) * 100

        # test model on CFP-FP
        if (args.eval_dataset == 'cfp') or (args.eval_dataset == 'all'):
            getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_cfpfp_result.mat'), net, device, cfpfpdataset, cfpfploader, qualnet=args.qualnet)
            cfp_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_cfpfp_result.mat'))
            print('Evaluation Result on CFP-ACC %dX - %.2f' %(down_size, np.mean(cfp_accs) * 100))
            result['cfp'][str(down_size)] = np.mean(cfp_accs) * 100
        
        # test model on LFW
        if (args.eval_dataset == 'lfw') or (args.eval_dataset == 'all'):
            getFeatureFromTorch(os.path.join(args.checkpoint_dir, 'result/cur_lfw_result.mat'), net, device, lfwdataset, lfwloader, qualnet=args.qualnet)
            lfw_accs = evaluation_10_fold(os.path.join(args.checkpoint_dir, 'result/cur_lfw_result.mat'))
            print('Evaluation Result on LFW-ACC %dX - %.2f' %(down_size, np.mean(lfw_accs) * 100))
            result['lfw'][str(down_size)] = np.mean(lfw_accs) * 100

        # Average
        if (args.eval_dataset == 'agedb30') or (args.eval_dataset == 'all'):
            average_age += np.mean(age_accs) * 100
        if (args.eval_dataset == 'cfp') or (args.eval_dataset == 'all'):
            average_cfp += np.mean(cfp_accs) * 100
        if (args.eval_dataset == 'lfw') or (args.eval_dataset == 'all'):
            average_lfw += np.mean(lfw_accs) * 100
        
        
    if len(eval_list) > 1:
        if (args.eval_dataset == 'agedb30') or (args.eval_dataset == 'all'):
            print('average - age_accs : %.2f' %(average_age / len(eval_list)))
            result['agedb30'][down_size] = average_age / len(eval_list)

        if (args.eval_dataset == 'cfp') or (args.eval_dataset == 'all'):
            print('average - cfp_accs : %.2f' %(average_cfp / len(eval_list)))
            result['cfp'][down_size] = average_cfp / len(eval_list)

        if (args.eval_dataset == 'lfw') or (args.eval_dataset == 'all'):
            print('average - lfw_accs : %.2f' %(average_lfw / len(eval_list)))
            result['lfw'][down_size] = average_lfw / len(eval_list)
    
    print('------------------------------------------------------------')
    with open(os.path.join(args.save_dir, args.prefix + '.pkl'), 'wb') as f:
        pickle.dump(result, f)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--checkpoint_dir', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/test/student-casia/iresnet50-E-IR-CosFace/resol1-random/F_SKD_CROSS-P{20.0,4.0}-hint-BN{True}', help='model save dir')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')

    parser.add_argument('--save_dir', type=str, default='imp/', help='result save dir')
    parser.add_argument('--prefix', type=str, default='aaa', help='prefix name')
    parser.add_argument('--eval_dataset', type=str, default='agedb30', help='save dataset')
    parser.add_argument('--eval_cross_resolution', type=lambda x: x.lower() == 'true', default=False)

    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gpus', type=str, default='2', help='model prefix')
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