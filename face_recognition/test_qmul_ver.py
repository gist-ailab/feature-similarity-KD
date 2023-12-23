import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backbone.iresnet import iresnet18, iresnet50
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
from evaluation.insightface_ijb_helper.dataloader import prepare_dataloader
from evaluation.insightface_ijb_helper import eval_helper as eval_helper_verification
import cv2

import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import roc_curve, auc
import pickle


class DATASET(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False):
        super(DATASET, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]

        img = cv2.imread(image_path)
        img = img[:, :, :3]

        img = cv2.resize(img, dsize=(112, 112))

        if self.image_is_saved_with_swapped_B_and_R:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx, os.path.basename(image_path)
    

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

def infer_images(model, image_list, batch_size, use_flip_test, qualnet=False):
    dataset = DATASET(image_list)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=4)
    model.eval()
    features = []
    pathes = []
    with torch.no_grad():
        for images, idx, path in tqdm(dataloader):
            feature = model(images.to("cuda"))
            if qualnet:
                feature = feature[0]

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda"))
                if qualnet:
                    flipped_feature = flipped_feature[0]

                fused_feature = (feature + flipped_feature) / 2
                features.append(fused_feature.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())

            pathes.append(path)

    features = np.concatenate(features, axis=0)
    pathes = np.concatenate(pathes, axis=0)

    img_feats = np.array(features).astype(np.float32)
    assert len(features) == len(image_list)
    return img_feats, pathes


def verification(features, path_list, pos_pair, neg_pair, save_dir, prefix, do_norm=True):
    if do_norm: 
        features = features / np.linalg.norm(features, ord=2, axis=1).reshape(-1,1)

    # Get Score
    score_list, label_list = [], []
    for pos_path in pos_pair:
        p_l = features[np.where(path_list == pos_path[0])[0]][0]
        p_r = features[np.where(path_list == pos_path[1])[0]][0]
        score = np.sum(p_l * p_r)
        score_list.append(score)
        label_list.append(1)

    for neg_path in neg_pair:
        n_l = features[np.where(path_list == neg_path[0])[0]][0]
        n_r = features[np.where(path_list == neg_path[1])[0]][0]
        score = np.sum(n_l * n_r)
        score_list.append(score)
        label_list.append(0)

    score_list, label_list = np.array(score_list), np.array(label_list)

    # Calculate
    fpr, tpr, _ = roc_curve(label_list, score_list)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    
    tpr_fpr_row = []
    tpr_fpr_row.append("QMUL-Surv-Verification")
    x_labels = [0.001, 0.01, 0.1, 0.3]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    
    with open(os.path.join(args.save_dir, args.prefix + '.pkl'), 'wb') as f:
        pickle.dump(tpr_fpr_row, f)
        
    print(tpr_fpr_row)


def load_model(args):
    device = torch.device('cuda')

    # define backbone and margin layer
    if args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)
    elif args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)

    # Load Pretrained Teacher
    net_ckpt = torch.load(os.path.join(args.checkpoint_path, 'last_net.ckpt'), map_location='cpu')['net_state_dict']
    new_ckpt = OrderedDict()
    for key, value in net_ckpt.items():
        if ('conv_bridge' not in key) and ('hint' not in key):
            new_ckpt[key] = value
    net.load_state_dict(new_ckpt, strict=False)

    net = net.to(device)
    net.eval()
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/QMUL-SurvFace/Face_Verification_Test_Set/')
    parser.add_argument('--gpus', default='7', type=str)
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--mode', type=str, default='ir', help='attention type')
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E') #
    parser.add_argument('--checkpoint_path', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/test/old_result_(m=default)/student-casia/iresnet50-E-IR-CosFace/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}/seed{5}', help='scale size')
    parser.add_argument('--save_dir', type=str, default='result/', help='scale size')
    parser.add_argument('--prefix', type=str, default='aa', help='scale size')

    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--qualnet', type=str2bool, default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Dataset
    print('use_flip_test', args.use_flip_test)
    os.makedirs(args.save_dir, exist_ok=True)

    model = load_model(args)

    # get features and fuse
    image_list = glob(os.path.join(args.data_dir, 'verification_images/*.jpg'))
    image_feats, pathes = infer_images(model=model,
                                        image_list=image_list,
                                        batch_size=args.batch_size,
                                        use_flip_test=args.use_flip_test,
                                        qualnet=args.qualnet)
    
    positive_pair = loadmat(os.path.join(args.data_dir, 'positive_pairs_names.mat'))['positive_pairs_names'] 
    negative_pair = loadmat(os.path.join(args.data_dir, 'negative_pairs_names.mat'))['negative_pairs_names'] 


    # run protocol
    # identification(args.data_dir, data_name, img_input_feats, save_dir=args.save_dir, aligned=args.aligned)
    verification(image_feats, pathes, positive_pair, negative_pair, save_dir=args.save_dir, prefix=args.prefix)