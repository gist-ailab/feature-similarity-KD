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


def identification(gallery_feature_list, mated_probe_feature_list, unmated_probe_feature_list,
                   gallery_sets, mated_probe_sets,  
                   save_dir, prefix, do_norm=True, template=True):
    # Feature and Pathes
    gallery_feature, gallery_feature_paths = gallery_feature_list
    mated_feature, mated_feature_paths = mated_probe_feature_list
    unmated_feature, unmated_feature_paths = unmated_probe_feature_list

    # ID and Pathes
    gallery_id, gallery_path = gallery_sets
    mated_id, mated_path = mated_probe_sets

    # Normalization
    if do_norm: 
        gallery_feature = gallery_feature / np.linalg.norm(gallery_feature, ord=2, axis=1).reshape(-1,1)
        mated_feature = mated_feature / np.linalg.norm(mated_feature, ord=2, axis=1).reshape(-1,1)
        unmated_feature = unmated_feature / np.linalg.norm(unmated_feature, ord=2, axis=1).reshape(-1,1)

    
    # Template Construction

    pass

    return None






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
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/QMUL-SurvFace/Face_Identification_Test_Set/')
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
    gallery_list = glob(os.path.join(args.data_dir, 'gallery/*.jpg'))
    mated_probe_list = glob(os.path.join(args.data_dir, 'mated_probe/*.jpg'))
    unmated_probe_list = glob(os.path.join(args.data_dir, 'unmated_probe/*.jpg'))

    gallery_feats, gallery_feat_pathes = infer_images(model=model,
                                        image_list=gallery_list,
                                        batch_size=args.batch_size,
                                        use_flip_test=args.use_flip_test,
                                        qualnet=args.qualnet)
    
    mated_probe_feats, mated_feat_pathes = infer_images(model=model,
                                        image_list=mated_probe_list,
                                        batch_size=args.batch_size,
                                        use_flip_test=args.use_flip_test,
                                        qualnet=args.qualnet)
    
    unmated_probe_feats, unmated_feat_pathes = infer_images(model=model,
                                        image_list=unmated_probe_list,
                                        batch_size=args.batch_size,
                                        use_flip_test=args.use_flip_test,
                                        qualnet=args.qualnet)
    
    g_mat = loadmat(os.path.join(args.data_dir, 'gallery_img_ID_pairs.mat'))
    gallery_ids, gallery_pathes = g_mat['gallery_ids'], g_mat['gallery_set']

    p_mat = loadmat(os.path.join(args.data_dir, 'mated_probe_img_ID_pairs.mat')) 
    mated_ids, mated_pathes = p_mat['mated_probe_ids'], p_mat['mated_probe_set']

    # run protocol
    # identification(args.data_dir, data_name, img_input_feats, save_dir=args.save_dir, aligned=args.aligned)
    identification([gallery_feats, gallery_feat_pathes], [mated_probe_feats, mated_feat_pathes], [unmated_probe_feats, unmated_feat_pathes],
                   [gallery_ids, gallery_pathes], [mated_ids, mated_pathes], save_dir=args.save_dir, prefix=args.prefix, do_norm=True, template=True)