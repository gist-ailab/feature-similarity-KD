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
import heapq
import math


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
    unmated_feature, _ = unmated_probe_feature_list

    # Normalization
    if do_norm: 
        gallery_feature = gallery_feature / np.linalg.norm(gallery_feature, ord=2, axis=1).reshape(-1,1)
        mated_feature = mated_feature / np.linalg.norm(mated_feature, ord=2, axis=1).reshape(-1,1)
        unmated_feature = unmated_feature / np.linalg.norm(unmated_feature, ord=2, axis=1).reshape(-1,1)

    # ID and Pathes
    gallery_id_list, gallery_path_list = gallery_sets
    
    mated_id_list, mated_path_list = mated_probe_sets
    mated_dict = {key[0][0]: value[0] for key, value in zip(mated_path_list, mated_id_list)}

    # Template Construction
    gallery_unique_id_list = np.unique(gallery_id_list)
    template_feat_list = []
    template_id_list = []
    for id in tqdm(gallery_unique_id_list):
        # append id
        template_id_list.append(id)

        # append feature
        path_id_list = gallery_path_list[np.where(gallery_id_list == id)[0], 0]
        for ix, path in enumerate(path_id_list):
            feat_ix = gallery_feature[np.where(gallery_feature_paths == path[0])[0]][0]
            if ix == 0:
                feature = feat_ix
            else:
                feature += feat_ix
        
        feature /= len(path_id_list)
        template_feat_list.append(feature)

    template_feat_list = np.array(template_feat_list)
    if do_norm: 
        template_feat_list = template_feat_list / np.linalg.norm(template_feat_list, ord=2, axis=1).reshape(-1,1)
    template_id_list = np.array(template_id_list)


    # Construct ID list
    probe_id_list = []
    for path in tqdm(mated_feature_paths):
        id_ix = mated_dict[path]
        probe_id_list.append(id_ix)
    probe_id_list += [-100] * len(unmated_feature)
    probe_id_list = np.array(probe_id_list)

    # Construct Probe Features
    probe_feats = np.concatenate([mated_feature, unmated_feature], axis=0)

    # Measure TPIR{N}@FPIR{M}
    result = evaluation(probe_id_list, probe_feats, template_feat_list, template_id_list)
    return result


def evaluation(query_ids, probe_feats, gallery_feats, gallery_ids):
    similarity = np.dot(probe_feats, gallery_feats.T)

    negative_index = np.where(query_ids == -100)[0]
    scores1 = np.empty(len(negative_index))
    for n in range(len(negative_index)):
        score = np.sort(similarity[negative_index[n], :])[::-1]
        scores1[n] = score[0]  # only consider the highest score for unmated probe

    # Searching step
    step = (np.max(scores1) - np.min(scores1)) / 1000
    thresholds = []
    FPIRs = []
    for threshold in np.arange(np.min(scores1), np.max(scores1), step):
        current_fpir = np.sum(scores1 >= threshold) / len(scores1)
        thresholds.append(threshold)
        FPIRs.append(current_fpir)


    # Compute FNIR corresponding to FPIR
    positive_index = np.where(query_ids != -100)[0]
    L = 20
    gt_scores = np.empty(len(positive_index))
    for p in range(len(positive_index)):
        query_id = query_ids[positive_index[p]]
        similarity_ix = similarity[positive_index[p], :]
        top_indices = np.argsort(similarity_ix)[::-1][:L]
        
        if query_id in gallery_ids[top_indices]:
            gt_scores[p] = similarity_ix[top_indices][np.where(gallery_ids[top_indices] == query_id)[0]]
        else:
            gt_scores[p] = -100000

    FNIRs = []
    for threshold in thresholds:
        current_fnir = np.sum(gt_scores < threshold) / len(gt_scores)
        FNIRs.append(current_fnir)


    # Sort FPIRs and corresponding FNIRs and thresholds
    FPIRs, fpir_index = np.sort(FPIRs), np.argsort(FPIRs)
    FNIRs = np.array(FNIRs)[fpir_index]
    thresholds = np.array(thresholds)[fpir_index]

    # Draw the TPIR@FPIR curve
    # plt.figure()
    # plt.plot(FPIRs, 1 - np.array(FNIRs))
    # plt.title('TPIR@FPIR Curve')
    # plt.xlabel('FPIR')
    # plt.ylabel('TPIR')

    # Show the TPIR20@FPIR=0.3/0.2/0.1 results & AUC
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    FPIRs_03_idx, _ = find_nearest(FPIRs, 0.3)
    FPIRs_02_idx, _ = find_nearest(FPIRs, 0.2)
    FPIRs_01_idx, _ = find_nearest(FPIRs, 0.1)
    TPIR_FPIR_03 = 1 - FNIRs[FPIRs_03_idx]
    TPIR_FPIR_02 = 1 - FNIRs[FPIRs_02_idx]
    TPIR_FPIR_01 = 1 - FNIRs[FPIRs_01_idx]
    AUC = np.trapz(1 - np.array(FNIRs), FPIRs)

    print(f'TPIR20@FPIR=0.3/0.2/0.1: {TPIR_FPIR_03} / {TPIR_FPIR_02} / {TPIR_FPIR_01}')
    print(f'AUC: {AUC}')
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
    net.load_state_dict(new_ckpt, strict=True)

    net = net.to(device)
    net.eval()
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do qmul test')
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/QMUL-SurvFace/Face_Identification_Test_Set/')
    parser.add_argument('--gpus', default='7', type=str)
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--mode', type=str, default='ir', help='attention type')
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E') #
    parser.add_argument('--checkpoint_path', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/case1/HR-LR-PHOTO{0.2},LR{0.2},type{range}/iresnet50-AdaFace-0.4', help='scale size')
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