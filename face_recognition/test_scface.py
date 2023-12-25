import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backbone.iresnet import iresnet18, iresnet50
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
from evaluation.insightface_ijb_helper.dataloader import prepare_dataloader
from evaluation.insightface_ijb_helper import eval_helper_identification

import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from glob import glob
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SCFACE_DATASET(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False):
        super(SCFACE_DATASET, self).__init__()

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
        img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_LINEAR)

        if self.image_is_saved_with_swapped_B_and_R:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def infer_images(model, image_list, landmark_list_path, batch_size, use_flip_test, qualnet=False):
    dataset = SCFACE_DATASET(image_list)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=4)
    model.eval()
    features = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
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

    features = np.concatenate(features, axis=0)
    return features


def load_model(args):
    device = torch.device('cuda')

    # define backbone and margin layer
    if args.backbone == 'iresnet18':
        net = iresnet18(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)
    elif args.backbone == 'iresnet50':
        net = iresnet50(attention_type=args.mode, pooling=args.pooling, qualnet=args.qualnet)

    # Load Pretrained Teacher
    net_ckpt = torch.load(os.path.join(args.checkpoint_path), map_location='cpu')['net_state_dict']
    new_ckpt = OrderedDict()
    for key, value in net_ckpt.items():
        if ('conv_bridge' not in key) and ('hint' not in key):
            new_ckpt[key] = value
    net.load_state_dict(new_ckpt, strict=True)

    net = net.to(device)
    net.eval()
    return net

def calc_accuracy(probe_feats, p_l, gallery_feats, g_l, dist, save_dir, do_norm=True):
    if do_norm: 
        probe_feats = probe_feats / np.linalg.norm(probe_feats, ord=2, axis=1).reshape(-1,1)
        gallery_feats =  gallery_feats / np.linalg.norm(gallery_feats, ord=2, axis=1).reshape(-1,1)
        
    # Similarity
    result = (probe_feats @ gallery_feats.T)
    index = np.argsort(-result, axis=1)
    
    acc_list = []
    for rank in [1, 5, 10, 20]:
        correct = 0
        for ix, probe_label in enumerate(p_l):
            pred_label = g_l[index[ix][:rank]]
            
            if probe_label in pred_label:
                correct += 1
                
        acc = correct / len(p_l)
        acc_list += [acc * 100]
    
    print(acc_list)
    pd.DataFrame({'rank':[1, 5, 10, 20], 'values':acc_list}).to_csv(os.path.join(save_dir, 'scface_dist%d_not_result.csv' %(dist)), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/scface_chen/SCFace_MTCNN/test_80')
    parser.add_argument('--gpus', default='7', type=str)
    parser.add_argument('--batch_size', default=512, type=int, help='')
    parser.add_argument('--mode', type=str, default='ir', help='attention type')
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E') #
    parser.add_argument('--checkpoint_path', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/face_recognition/checkpoint/case1/HR-LR-PHOTO{0.2},LR{0.2},type{range}/iresnet50-AdaFace-0.4/last_net.ckpt', help='scale size')
    parser.add_argument('--save_dir', type=str, default='imp/', help='scale size')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--qualnet', type=str2bool, default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Dataset
    os.makedirs(args.save_dir, exist_ok=True)
    model = load_model(args)
    
    # get features and fuse
    gallery_list = glob(os.path.join(args.data_dir, 'gallery/*/*.jpg'))
    gallery_labels = np.array([int(os.path.basename(gallery_path).split('_')[0]) for gallery_path in gallery_list])
    gallery_feats = infer_images(model=model,
                               image_list=gallery_list,
                               landmark_list_path=None,
                               batch_size=args.batch_size,
                               use_flip_test=args.use_flip_test,
                               qualnet=args.qualnet)


    # Evaluation for 3 Distances
    for distance in [1,2,3]:
        probe_list = glob(os.path.join(args.data_dir, 'probe_d%d/*/*.jpg') %distance)
        probe_labels = np.array([int(os.path.basename(probe_path).split('_')[0]) for probe_path in probe_list])

        # run protocol
        probe_feats = infer_images(model=model,
                                    image_list=probe_list,
                                    landmark_list_path=None,
                                    batch_size=args.batch_size,
                                    use_flip_test=args.use_flip_test,
                                    qualnet=args.qualnet)

        # Identification
        calc_accuracy(probe_feats, probe_labels, gallery_feats, gallery_labels, distance, args.save_dir, do_norm=True)