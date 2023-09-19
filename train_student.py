#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import torch.utils.data
from backbone.iresnet import iresnet18, iresnet50
from torch.nn import DataParallel
from margin.ArcMarginProduct import ArcMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.AdaMarginProduct import AdaMarginProduct
from utility.log import init_log
from dataset.casia_webface import CASIAWebFace
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
from metric.distill_loss import cosine_loss, normalize, cross_kd, RKD_cri, AT_cri, mse_loss
from collections import OrderedDict
from utility.hook import feature_hook


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


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, args.down_size, transform=transform, equal=args.equal, interpolation_option=args.interpolation, cross_sampling=args.cross_sampling)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    

    # define backbone and margin layer
    if args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling)
    elif args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling)


    # Margin
    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'AdaFace':
        margin = AdaMarginProduct(args.feature_dim, trainset.class_nums)
    else:
        print(args.margin_type, 'is not available!')


    # Load Teacher Model and Freeze
    aux_net = iresnet50(attention_type=args.mode, pooling=args.pooling)
    aux_margin = deepcopy(margin)

    
    # Load Pretrained Teacher
    net_ckpt = torch.load(args.teacher_path, map_location='cpu')['net_state_dict']
    
    new_ckpt = OrderedDict()
    for key, value in net_ckpt.items():
        if 'conv_bridge' not in key:
            new_ckpt[key] = value
    
    aux_net.load_state_dict(new_ckpt)
    aux_margin.load_state_dict(torch.load(args.teacher_path.replace('_net.ckpt', '_margin.ckpt'), map_location='cpu')['net_state_dict'])
    
    for param in aux_net.parameters():
        param.requires_grad = False
    
    for param in aux_margin.parameters():
        param.requires_grad = False


    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[18000, 28000, 36000, 44000], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        aux_net = DataParallel(aux_net).to(device)
        margin = DataParallel(margin).to(device)
        aux_margin = DataParallel(aux_margin).to(device)
            
    else:
        net = net.to(device)
        aux_net = aux_net.to(device)
        margin = margin.to(device)
        aux_margin = aux_margin.to(device)

    total_iters = 0


    if args.distill_type == 'A_SKD':
        target_layer = 'attention_target'
        HR_manager = feature_hook(aux_net, multi_gpu=multi_gpus, target_layer=target_layer)    
        LR_manager = feature_hook(net, multi_gpu=multi_gpus, target_layer=target_layer)
        hook = True
    else:
        LR_manager = None
        HR_manager = None
        hook = False
    

    # Run
    GOING = True
    while GOING:
        # train model and freeze batchnorm for main network
        net.train()        
        margin.train()
        aux_net.eval()
        aux_margin.eval()

        since = time.time()
        for data in tqdm(trainloader):            
            if args.cross_sampling:
                HR_img, LR_img, HR_pos_img, LR_pos_img, label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            else:
                HR_img, LR_img, label = data[0].to(device), data[1].to(device), data[2].to(device)                  
            
            
            # Clear Hook
            if hook:
                LR_manager.attention = []
                HR_manager.attention = []
                
                
            # High-Resolution Forward
            with torch.no_grad():
                HR_logits, HR_feat_list = aux_net(HR_img, extract_feature=True)
                if args.margin_type == 'AdaFace':
                    HR_norm = torch.norm(HR_logits, 2, 1, True)
                    HR_out = margin(HR_logits, HR_norm, label)
                else:
                    HR_out = aux_margin(HR_logits, label)
            
            # Low-Resolution Forward     
            LR_logits, LR_feat_list = net(LR_img, extract_feature=True)
            if args.margin_type == 'AdaFace':
                LR_norm = torch.norm(LR_logits, 2, 1, True)
                LR_out = margin(LR_logits, LR_norm, label)
            else:
                LR_out = margin(LR_logits, label)
            
            
            # multi-gpu settings
            if multi_gpus and hook:
                HR_imp, LR_imp = [], []

                hr_device = np.array([str(hr[0].device) for hr in HR_manager.attention])
                lr_device = np.array([str(lr[0].device) for lr in LR_manager.attention])
                
                for ord in np.unique(hr_device):
                    for ix in np.where(hr_device == ord)[0]:
                        HR_imp.append(HR_manager.attention[ix])
                    for ix in np.where(lr_device == ord)[0]:
                        LR_imp.append(LR_manager.attention[ix])
                
                HR_manager.attention = HR_imp
                LR_manager.attention = LR_imp
                del HR_imp, LR_imp
            
            
            # # positive sampling            
            # if args.cross_sampling:
            #     with torch.no_grad():
            #         _, HR_pos_feat_list = aux_net(HR_pos_img, extract_feature=True)
            #     _, LR_pos_feat_list = net(LR_pos_img, extract_feature=True)
                

            # Sample choice for distillation            
            # _, predict = torch.max(HR_out.data, 1)
            # correct_index = (predict == label)
            
            B = HR_img.size(0)
            correct_index = torch.ones(B).bool()
                
            # Recognition Loss
            cri_loss = criterion(LR_out, label)
            
            # Distillation
            distill_param = list(map(float, args.distill_param.split(',')))

            # Point distillation
            point_loss = 0.
            if distill_param[0] > 0.:
                if 'F_SKD' in args.distill_type:
                    for HR_feat, LR_feat in zip(HR_feat_list, LR_feat_list):
                        point_loss += cosine_loss(LR_feat[correct_index], HR_feat[correct_index]) / len(HR_feat_list)
                
                elif args.distill_type == 'A_SKD' : 
                    for HR_feat, LR_feat in zip(HR_manager.attention, LR_manager.attention):
                        point_loss += (cosine_loss(LR_feat[0][correct_index], HR_feat[0][correct_index]) + cosine_loss(LR_feat[1][correct_index], HR_feat[1][correct_index])) / len(HR_manager.attention) / 2
                
                elif args.distill_type == 'RKD':
                    point_loss += RKD_cri(w_dist=1., w_angle=2.)(LR_logits, HR_logits)
                    
                elif args.distill_type == 'FitNet':
                    for HR_feat, LR_feat in zip(HR_feat_list, LR_feat_list):
                        point_loss += mse_loss(LR_feat[correct_index], HR_feat[correct_index]) / len(HR_feat_list)
                
                elif args.distill_type == 'AT':
                    for HR_feat, LR_feat in zip(HR_feat_list, LR_feat_list):
                        point_loss += AT_cri(p=2)(LR_feat[correct_index], HR_feat[correct_index]) / len(HR_feat_list)
            
                else:
                    raise('No Proper Distillation')
                

            # Cross Distillation Loss
            cross_loss = 0.    
            if distill_param[1] > 0.:
                # cross_loss += cross_kd()(LR_feat_list[-1][correct_index], LR_pos_feat_list[-1][correct_index], HR_feat_list[-1][correct_index], HR_pos_feat_list[-1][correct_index]) 
                
                new_correct_index = list(range(torch.sum(correct_index).item()))
                random.shuffle(new_correct_index)
                cross_loss += cross_kd()(LR_feat_list[-1][correct_index], LR_feat_list[-1][correct_index][new_correct_index], HR_feat_list[-1][correct_index], HR_feat_list[-1][correct_index][new_correct_index])    
                
        
            distill_loss = point_loss * distill_param[0] + cross_loss * distill_param[1]
            
            # Total Loss
            total_loss = cri_loss + distill_loss

            # Optim
            optimizer_ft.zero_grad()
            total_loss.backward()
            optimizer_ft.step()


            # Clear Features
            if hook:
                HR_manager.attention = []
                LR_manager.attention = []
                
            
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(LR_out.data, 1)
                total = label.size(0)

                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                _print("Iters: {:0>6d}, cri_loss: {:.4f}, distill_loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, cri_loss.item(), distill_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            
            exp_lr_scheduler.step()
            
            # Stop
            if total_iters == 47000:
                GOING = False
                break
        
            # Next Iterations
            total_iters += 1
            
    

    # Remove Hook
    if hook:
        HR_manager.remove_hook()
        LR_manager.remove_hook()  
        
    
    # Save Last Epoch
    msg = 'Saving checkpoint: {}'.format(total_iters)
    _print(msg)
    if multi_gpus:
        net_state_dict = net.module.state_dict()
        margin_state_dict = margin.module.state_dict()
    else:
        net_state_dict = net.state_dict()
        margin_state_dict = margin.state_dict()
        
    torch.save({
        'iters': total_iters,
        'net_state_dict': net_state_dict},
        os.path.join(save_dir, 'last_net.ckpt'))
    torch.save({
        'iters': total_iters,
        'net_state_dict': margin_state_dict},
        os.path.join(save_dir, 'last_margin.ckpt'))


    # test dataset
    net.eval()
    margin.eval()
    
    print('Evaluation on LFW, AgeDB-30. CFP')
    os.makedirs(os.path.join(args.save_dir, 'result'), exist_ok=True)
    
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
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader)
        age_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'))
        _print('Evaluation Result on AgeDB-30 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))

        # test model on CFP-FP
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'), net, device, cfpfpdataset, cfpfploader)
        cfp_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'))
        _print('Evaluation Result on CFP-ACC %dX - %.2f' %(down_size, np.mean(cfp_accs) * 100))
        
        # test model on LFW
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'), net, device, lfwdataset, lfwloader)
        lfw_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'))
        _print('Evaluation Result on LFW-ACC %dX - %.2f' %(down_size, np.mean(lfw_accs) * 100))

        # Average
        average_age += np.mean(age_accs) * 100
        average_cfp += np.mean(cfp_accs) * 100
        average_lfw += np.mean(lfw_accs) * 100
        
        
    if len(eval_list) > 1:
        _print('average - age_accs : %.2f' %(average_age / len(eval_list)))
        _print('average - cfp_accs : %.2f' %(average_cfp / len(eval_list)))
        _print('average - lfw_accs : %.2f' %(average_lfw / len(eval_list)))
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset/')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--save_dir', type=str, default='checkpoint/imp/', help='model save dir')
    parser.add_argument('--mode', type=str, default='ir', help='attention type', choices=['ir', 'cbam'])
    parser.add_argument('--backbone', type=str, default='iresnet50')
    
    parser.add_argument('--interpolation', type=str)
    parser.add_argument('--pooling', type=str, default='A')
    
    parser.add_argument('--margin_type', type=str, default='CosFace', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--equal', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--gpus', type=str, default='5', help='model prefix')
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--distill_param', type=str, default='1.0,1.0', help='hyperparams for distillation loss')
    parser.add_argument('--distill_type', type=str, default='F_SKD_BLOCK', help='distillation types')
    parser.add_argument('--teacher_path', type=str, default='checkpoint/teacher/iresnet50-A-IR/last_net.ckpt')
    args = parser.parse_args()

    # args.cross_sampling = 'cross' in args.distill_type.lower()
    args.cross_sampling = False
    

    # Path
    args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
    args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    args.lfw_test_root = os.path.join(args.data_dir, 'evaluation/lfw')
    args.lfw_file_list = os.path.join(args.data_dir, 'evaluation/lfw.txt')
    args.agedb_test_root = os.path.join(args.data_dir, 'evaluation/agedb_30')
    args.agedb_file_list = os.path.join(args.data_dir, 'evaluation/agedb_30.txt')
    args.cfpfp_test_root = os.path.join(args.data_dir, 'evaluation/cfp_fp')
    args.cfpfp_file_list = os.path.join(args.data_dir, 'evaluation/cfp_fp.txt')

    if args.down_size not in [0, 112]:
        assert (args.interpolation == 'random') or (args.interpolation == 'fix')

    # Seed
    set_random_seed(args.seed)
    
    # Run    
    train(args)