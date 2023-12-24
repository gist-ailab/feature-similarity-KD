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
from margin.MagMarginProduct import MagMarginProduct, MagLoss
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


def setup(args):
    # 1. setting for distributed training
    args.global_rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    if args.global_rank is not None and args.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(args.global_rank, args.local_rank))

    # 2. init_process_group
    dist.init_process_group(backend="nccl")
    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    return


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
    setup(args)

    # log init
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'result'), exist_ok=True)

    if args.local_rank == 0:
        logging = init_log(save_dir)
        _print = logging.info
    else:
        logging = None
        _print = None

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # train dataset
    args.batch_total_size = args.batch_size
    args.batch_size = int(args.batch_size / args.world_size)
    trainset = FaceDataset(args.train_root, args.dataset, args.train_file_list, args.down_size, transform=transform, photo_prob=args.photo_prob, lr_prob=args.lr_prob, size_type=args.size_type)

    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False, sampler=train_sampler)
    
    # define backbone and margin layer
    if args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling)
    elif args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling)    

    # Head
    if args.dataset == 'webface4m' or args.dataset == 'ms1mv2':
        scale = 64.0
    else:
        scale = 32.0

    # Head
    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, m=args.margin_float, s=scale)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums, m=args.margin_float, s=scale)
    elif args.margin_type == 'AdaFace':
        margin = AdaMarginProduct(args.feature_dim, trainset.class_nums, m=args.margin_float, s=scale)
    elif args.margin_type == 'MagFace':
        margin = MagMarginProduct(args.feature_dim, trainset.class_nums, m=args.margin_float, s=scale)
    else:
        print(args.margin_type, 'is not available!')

    
    # define optimizers for different layer
    if args.margin_type == 'MagFace':
        criterion = MagLoss().to(args.local_rank)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(args.local_rank)

    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)


    if args.dataset == 'casia':
        ratio = 512 / args.batch_total_size
        iter_size = int(959 * ratio)
        print('iter_size: ', iter_size)
        if args.margin_type == 'AdaFace':
            finish_iters = (iter_size * 26)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[iter_size * 12, iter_size * 20, iter_size * 24], gamma=0.1)
        else:
            finish_iters = (iter_size * 24)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[iter_size * 10, iter_size * 18, iter_size * 22], gamma=0.1)

    elif args.dataset == 'mini_casia':
        finish_iters = 24000
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[9000, 14000, 18000, 22000], gamma=0.1)

    elif args.dataset == 'webface4m':
        ratio = 1024 / args.batch_total_size
        iter_size = int(4136 * ratio)
        print('iter_size: ', iter_size)
        if args.margin_type == 'AdaFace':
            finish_iters = (iter_size * 26)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[iter_size * 12, iter_size * 20, iter_size * 24], gamma=0.1)
        else:
            finish_iters = (iter_size * 24)
            exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[iter_size * 10, iter_size * 18, iter_size * 22], gamma=0.1)

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


    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    margin = torch.nn.SyncBatchNorm.convert_sync_batchnorm(margin)
    
    net = DDP(net.to(args.local_rank), device_ids=[args.local_rank])
    margin = DDP(margin.to(args.local_rank), device_ids=[args.local_rank])

    total_iters = 0

    # Run
    GOING = True
    while GOING:
        # train model
        net.train()
        margin.train()

        since = time.time()

        # trainloader.dataset.update_candidate()
        for data in tqdm(trainloader):            
            img, label = data[1].to(args.local_rank), data[2].to(args.local_rank)                  
                        
            raw_logits = net(img)
            if args.margin_type == 'AdaFace':
                norm = torch.norm(raw_logits, 2, 1, True)
                out = margin(raw_logits, norm, label)
            elif args.margin_type == 'MagFace':
                out, norm = margin(raw_logits)
            else:
                out = margin(raw_logits, label)
            
            # Loss
            if args.margin_type == 'MagFace':
                cri_loss, loss_g, out = criterion(out, label, norm)
                total_loss = cri_loss + loss_g * 20.0
            else:
                cri_loss = criterion(out, label)
                total_loss = cri_loss


            # Optim
            optimizer_ft.zero_grad()
            total_loss.backward()
            optimizer_ft.step()

            # print train information
            if (total_iters % 100 == 0) and (args.local_rank==0):
                # current training accuracy
                _, predict = torch.max(out.data, 1)
                total = label.size(0)

                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                _print("Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))


            # save model
            if (total_iters % args.save_freq == 0) and (args.local_rank==0):
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                net_state_dict = net.module.state_dict()
                margin_state_dict = margin.module.state_dict()
                    
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
            
            
    # Evaluation 
    if args.local_rank == 0:
        msg = 'Saving checkpoint: {}'.format(total_iters)
        _print(msg)

        net_state_dict = net.module.state_dict()
        margin_state_dict = margin.module.state_dict()
            
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
            
        result_dict = {}

        if 'mini' in args.dataset:
            average_mini = 0.
            for down_size in eval_list:
                valdataset =  FaceDataset(args.train_root, args.dataset, args.val_file_list, down_size, transform=transform, 
                                          teacher_folder='', cross_sampling=False, margin=0.0)
                valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

                correct, total = 0, 0
                for data in tqdm(valloader):            
                    LR_img, label = data[1].to(args.local_rank), data[2].to(args.local_rank)
                    
                    with torch.no_grad():
                        LR_logits, LR_feat_list = net(LR_img, extract_feature=True)

                        if args.margin_type == 'AdaFace':
                            LR_norm = torch.norm(LR_logits, 2, 1, True)
                            LR_out = margin(LR_logits, LR_norm, label)
                        elif args.margin_type == 'MagFace':
                            LR_out, LR_norm = margin(LR_logits)    
                        else:
                            LR_out = margin(LR_logits, label)

                    # current training accuracy
                    _, predict = torch.max(LR_out.data, 1)
                    
                    total += label.size(0)
                    correct += (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                
                result_dict['%s_%dX' %(args.dataset, down_size)] = float(100 * correct / total)
                _print('Evaluation Result on %s %dX - %.2f' %(args.dataset, down_size, 100 * correct / total))
                average_mini += 100 * correct / total

            if len(eval_list) > 1:
                result_dict['%s_avg' %args.dataset] = float(average_mini / len(eval_list))
                _print('average - %s : %.2f' %(args.dataset, average_mini / len(eval_list)))  

            np.savez(os.path.join(args.save_dir, 'result', 'mini_result.npz'), **result_dict)
            
        else:
            for cross_resolution in [False, True]:
                average_age = 0.
                average_cfp = 0.
                average_lfw = 0.

                if cross_resolution:
                    _print('Cross Resolution Evaluation')
                else:
                    _print('Single Resolution Evaluation')

                for down_size in eval_list:
                    agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
                    cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
                    lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, down_size, transform=transform, cross_resolution=cross_resolution)
                    
                    agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
                    cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
                    lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

                    # test model on AgeDB30
                    getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'), net, args.local_rank, agedbdataset, agedbloader)
                    age_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'))
                    _print('Evaluation Result on AgeDB-30 %dX - %.2f' %(down_size, np.mean(age_accs) * 100))

                    if cross_resolution:
                        result_dict['agedb_cross_%dX' %down_size] = float(np.mean(age_accs) * 100)
                    else:
                        result_dict['agedb_%dX' %down_size] = float(np.mean(age_accs) * 100)

                    # test model on CFP-FP
                    getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'), net, args.local_rank, cfpfpdataset, cfpfploader)
                    cfp_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'))
                    _print('Evaluation Result on CFP-ACC %dX - %.2f' %(down_size, np.mean(cfp_accs) * 100))

                    if cross_resolution:
                        result_dict['cfpfp_cross_%dX' %down_size] = float(np.mean(cfp_accs) * 100)
                    else:
                        result_dict['cfpfp_%dX' %down_size] = float(np.mean(cfp_accs) * 100)

                    # test model on LFW
                    getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'), net, args.local_rank, lfwdataset, lfwloader)
                    lfw_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'))
                    _print('Evaluation Result on LFW-ACC %dX - %.2f' %(down_size, np.mean(lfw_accs) * 100))

                    if cross_resolution:
                        result_dict['lfw_cross_%dX' %down_size] = float(np.mean(lfw_accs) * 100)
                    else:
                        result_dict['lfw_%dX' %down_size] = float(np.mean(lfw_accs) * 100)

                    # Average
                    average_age += np.mean(age_accs) * 100
                    average_cfp += np.mean(cfp_accs) * 100
                    average_lfw += np.mean(lfw_accs) * 100
                    
                    
                if len(eval_list) > 1:
                    _print('average - age_accs : %.2f' %(average_age / len(eval_list)))
                    _print('average - cfp_accs : %.2f' %(average_cfp / len(eval_list)))
                    _print('average - lfw_accs : %.2f' %(average_lfw / len(eval_list)))

                    if cross_resolution:
                        result_dict['agedb_cross_avg'] = float(average_age / len(eval_list))
                        result_dict['cfpfp_cross_avg'] = float(average_cfp / len(eval_list))
                        result_dict['lfw_cross_avg'] = float(average_lfw / len(eval_list))
                    else:
                        result_dict['agedb_avg'] = float(average_age / len(eval_list))
                        result_dict['cfpfp_avg'] = float(average_cfp / len(eval_list))
                        result_dict['lfw_avg'] = float(average_lfw / len(eval_list))

                _print('------------------------------------------------------------')
            
            np.savez(os.path.join(args.save_dir, 'result', 'main_result.npz'), **result_dict)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--dataset', type=str, default='casia')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/SSDb/sung/dataset/face_dset')
    parser.add_argument('--save_dir', type=str, default='imp/', help='model save dir')
    parser.add_argument('--down_size', type=int, default=1) # 1 : all type, 0 : high, others : low
    parser.add_argument('--pooling', type=str, default='E') #

    parser.add_argument('--size_type', type=str, choices=['range', 'fix']) #
    parser.add_argument('--photo_prob', type=float)
    parser.add_argument('--lr_prob', type=float)

    parser.add_argument('--margin_float', type=float)

    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--margin_type', type=str, default='CosFace', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--mode', type=str, default='ir')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    args = parser.parse_args()


    # Path
    if args.dataset == 'casia':
        args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')

    elif args.dataset == 'webface4m':
        args.train_root = os.path.join(args.data_dir, 'webface4m/image')
        args.train_file_list = os.path.join(args.data_dir, 'webface4m/train.list')

    elif args.dataset == 'mini_casia':
        args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
        args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train_mini.list')
        args.val_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/val_mini.list')

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
    
    
    # Seed
    set_random_seed(args.seed)

    # Run
    train(args)