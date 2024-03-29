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
from dataset.svhn import SVHN, SVHN_CROSS
from tqdm import tqdm
from backbone.vgg import vgg8
from metric.distill_loss import cosine_loss, cross_sample_kd


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--resolution', default=8, type=int)
parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/svhn', type=str)
parser.add_argument('--teacher_path', default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/classification/checkpoint/teacher/teacher_epoch200_lr{0.1}/checkpoint.th', type=str)

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    teacher_model = torch.nn.DataParallel(resnet.__dict__['resnet56']())
    student_model = torch.nn.DataParallel(resnet.__dict__['resnet20']())

    teacher_model.cuda()
    student_model.cuda()

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # optionally resume from a checkpoint
    teacher_checkpoint = torch.load(args.teacher_path)
    teacher_model.load_state_dict(teacher_checkpoint['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        SVHN_CROSS(
                root=args.data_dir, split='train',
                pre_transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomCrop(32, 4)
                          ]),
                transform=transforms.Compose([
                          transforms.ToTensor(),
                          normalize,
                        ]), 
                download=False,
                resolution=args.resolution,
                cross_dict_path=os.path.join(os.path.dirname(args.teacher_path), 'cross_dict.pkl')
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # Evaluation Set
    eval_resolution = 8
    val_loader = torch.utils.data.DataLoader(
        SVHN(root=args.data_dir, split='test', 
            pre_transform=None,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ]),
        resolution=eval_resolution
        ),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], last_epoch=args.start_epoch - 1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, student_model, criterion) 
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, teacher_model, student_model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, student_model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': student_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, teacher_model, student_model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    teacher_model.eval()
    student_model.train()

    end = time.time()
    for i, data_ix in enumerate(tqdm(train_loader)):
        HR_input, LR_input, HR_pos_input, LR_pos_input, correct_index, target = data_ix
        correct_index = correct_index.bool()

        # Load Input
        data_time.update(time.time() - end)

        target = target.cuda()
        HR_input, LR_input = HR_input.cuda(), LR_input.cuda()
        HR_pos_input, LR_pos_input = HR_pos_input.cuda(), LR_pos_input.cuda()
        target_var = target

        # Comput Output
        with torch.no_grad():
            _, HR_feat_list, _ = teacher_model(HR_input, extract_feat=True)
            _, HR_pos_feat_list, _ = teacher_model(HR_pos_input, extract_feat=True)

        LR_output, LR_feat_list, _ = student_model(LR_input, extract_feat=True)
        _, LR_pos_feat_list, _ = student_model(LR_pos_input, extract_feat=True)

        # 1st-order KD loss
        point_loss = 0.
        for HR_feat, LR_feat in zip(HR_feat_list, LR_feat_list):
            point_loss += cosine_loss(LR_feat, HR_feat) / len(HR_feat_list)

        # 2nd-order KD loss
        cross_loss = cross_sample_kd()(LR_feat_list[-1][correct_index], LR_pos_feat_list[-1][correct_index],
                                        HR_feat_list[-1][correct_index], HR_pos_feat_list[-1][correct_index])       

        loss_distill = point_loss * 2.0 + cross_loss * 0.4
        loss_cri = criterion(LR_output, target_var)
        loss = loss_distill + loss_cri

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LR_output = LR_output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(LR_output.data, target)[0]
        losses.update(loss.item(), LR_input.size(0))
        top1.update(prec1.item(), LR_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (_, input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()