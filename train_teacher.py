#!/usr/bin/env python
# encoding: utf-8
import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

import torch.utils.data
from torch.nn import DataParallel
from backbone.iresnet import iresnet18, iresnet50
from margin.ArcMarginProduct import ArcMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.AdaMarginProduct import AdaMarginProduct

from utility.log import init_log
from dataset.train_dataset import FaceDataset
from dataset.agedb import AgeDB30
from dataset.cfp import CFP_FP
from dataset.lfw import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from evaluation.eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
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
    trainset = FaceDataset(args.train_root, args.dataset, args.train_file_list, args.down_size, transform=transform, equal=args.equal, interpolation_option=args.interpolation)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    # define backbone and margin layer
    if args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling)
    elif args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling)    
        
    # Head
    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'AdaFace':
        margin = AdaMarginProduct(args.feature_dim, trainset.class_nums)
    else:
        print(args.margin_type, 'is not available!')

    
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)


    if args.dataset == 'casia':
        finish_iters = 47000
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[18000, 28000, 36000, 44000], gamma=0.1)
    
    elif args.dataset == 'vggface':
        if args.margin_type == 'AdaFace':
            finish_iters = (6128 * 26)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6128 * 12, 6128 * 20, 6128 * 24], gamma=0.1)
        else:
            finish_iters = (6128 * 24)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6128 * 10, 6128 * 18, 6128 * 22], gamma=0.1)
    
    elif args.dataset == 'ms1mv2':
        if args.margin_type == 'AdaFace':
            finish_iters = (11373 * 26)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[11373 * 12, 11373 * 20, 11373 * 24], gamma=0.1)
        else:
            finish_iters = (11373 * 24)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[11373 * 10, 11373 * 18, 11373 * 22], gamma=0.1)


    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)

    total_iters = 0

    # Run
    GOING = True
    while GOING:
        # train model
        net.train()
        margin.train()

        since = time.time()
        for data in tqdm(trainloader):            
            img, label = data[1].to(device), data[2].to(device)                  
                        
            raw_logits = net(img)
            if args.margin_type == 'AdaFace':
                norm = torch.norm(raw_logits, 2, 1, True)
                out = margin(raw_logits, norm, label)
            else:
                out = margin(raw_logits, label)
            
            # Loss
            cri_loss = criterion(out, label)
            total_loss = cri_loss


            # Optim
            optimizer_ft.zero_grad()
            total_loss.backward()
            optimizer_ft.step()

            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(out.data, 1)
                total = label.size(0)

                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                _print("Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))


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
                    'net_state_dict': net_state_dict
                    },
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            
            exp_lr_scheduler.step()
            
            # Stop
            if total_iters == finish_iters:
                GOING = False
                break
        
            # Next Iterations
            total_iters += 1
            
            

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
        'net_state_dict': net_state_dict
        },
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
    parser.add_argument('--dataset', type=str, default='vggface')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset')
    parser.add_argument('--save_dir', type=str, default='imp/', help='model save dir')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--interpolation', type=str, default='random') # 
    parser.add_argument('--pooling', type=str, default='E') #

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--margin_type', type=str, default='CosFace', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--mode', type=str, default='ir')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--equal', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--gpus', type=str, default='5', help='model prefix')
    args = parser.parse_args()


    # Path
    if args.dataset == 'casia':
        args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    elif args.dataset == 'ms1mv2':
        args.train_root = os.path.join(args.data_dir, 'faces_emore/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_emore/train.list')
    elif args.dataset == 'vggface':
        args.train_root = os.path.join(args.data_dir, 'faces_vgg_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_vgg_112x112/train.list')
    else:
        raise('Select Proper Dataset')

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