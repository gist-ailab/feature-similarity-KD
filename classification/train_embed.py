import argparse
import os
os.chdir
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import backbone.resnet as resnet
from dataset.svhn import SVHN
from tqdm import tqdm
from backbone.vgg import vgg8
import pickle

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/svhn', type=str)
parser.add_argument('--teacher_path', default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/classification/checkpoint/teacher/teacher_epoch200_lr{0.1}/checkpoint.th', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    args.save_dir = os.path.dirname(args.teacher_path)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if 'resnet' in args.arch:
        model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    elif args.arch == 'vgg8':
        model = torch.nn.DataParallel(vgg8(num_classes=10))
    else:
        raise('Error!')
    model.cuda()

    teacher_checkpoint = torch.load(args.teacher_path)
    model.load_state_dict(teacher_checkpoint['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        SVHN(
                root=args.data_dir, split='train',
                pre_transform=None,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]),
                download=False,
                resolution=0
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # switch to train mode
    model.eval()
    feat_list, label_list = [], []
    for i, (input, _, target) in enumerate(tqdm(train_loader)):
        target = target.cuda()
        input_var = input.cuda()

        # compute output
        with torch.no_grad():
            _, _, feat = model(input_var, extract_feat=True)
        
        feat_list.append(feat)
        label_list.append(target)
    
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

    with open(os.path.join(args.save_dir, 'cross_dict.pkl'), 'wb') as f:
        pickle.dump(result, f)    
        


if __name__ == '__main__':
    main()