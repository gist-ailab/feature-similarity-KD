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


def infer_images(model, image_list, landmark_list_path, batch_size, use_flip_test, qualnet=False, aligned=True):
    landmarks = None
    aligned = False
    dataloader = prepare_dataloader(image_list, landmarks, batch_size, num_workers=0, image_size=(112,112), aligned=aligned)

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
    img_feats = np.array(features).astype(np.float32)
    assert len(features) == len(image_list)
    return img_feats


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
    net.load_state_dict(new_ckpt, strict=False)

    net = net.to(device)
    net.eval()
    return net

def calc_accuracy(probe_feats, probe_labels, gallery_feats, gallery_labels, dist, save_dir, aligned, do_norm=True):
    if do_norm: 
        probe_feats = probe_feats / np.linalg.norm(probe_feats, ord=2, axis=1).reshape(-1,1)
        gallery_feats =  gallery_feats / np.linalg.norm(gallery_feats, ord=2, axis=1).reshape(-1,1)
        
    # Similarity
    result = (probe_feats @ gallery_feats.T)
    index = np.argsort(-result, axis=1)
    
    p_l = np.array(probe_labels)
    g_l = np.array(gallery_labels)
    
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

    if aligned:
        pd.DataFrame({'rank':[1, 5, 10, 20], 'values':acc_list}).to_csv(os.path.join(save_dir, 'scface_dist%d_aligned_result.csv' %(dist)), index=False)
    else:
        pd.DataFrame({'rank':[1, 5, 10, 20], 'values':acc_list}).to_csv(os.path.join(save_dir, 'scface_dist%d_not_result.csv' %(dist)), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/scface/crop')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--batch_size', default=512, type=int, help='')
    parser.add_argument('--mode', type=str, default='ir', help='attention type')
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E') #
    parser.add_argument('--checkpoint_path', type=str, default='/home/jovyan/SSDb/sung/src/feature-similarity-KD/checkpoint/student-casia/iresnet50-E-IR-CosFace/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}/seed{5}/last_net.ckpt', help='scale size')
    parser.add_argument('--save_dir', type=str, default='imp/', help='scale size')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--qualnet', type=str2bool, default='False')
    parser.add_argument('--aligned', type=str2bool, default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Dataset
    os.makedirs(args.save_dir, exist_ok=True)

    model = load_model(args)
    
    person_list = np.random.choice(list(range(1,131)), 50, replace=False)

    # get features and fuse
    gallery_list = [os.path.join(args.data_dir, 'gallery/%03d_frontal.JPG.jpg' %person_id) for person_id in person_list]
    gallery_labels = np.array([int(os.path.basename(gallery_path).split('_')[0]) for gallery_path in gallery_list])
    gallery_feats = infer_images(model=model,
                               image_list=gallery_list,
                               landmark_list_path=None,
                               batch_size=args.batch_size,
                               use_flip_test=args.use_flip_test,
                               qualnet=args.qualnet,
                               aligned=args.aligned)


    # Evaluation for 3 Distances
    for distance in [1, 2, 3]:
        cam_list = [1,2,3,4,5]
        basename_list = ['%03d_cam%d_%d.jpg' %(person_id, cam_id, distance) for person_id in person_list for cam_id in cam_list]
        probe_list = [os.path.join(args.data_dir, 'probe', basename) for basename in basename_list]
        probe_labels = np.array([int(os.path.basename(probe_path).split('_')[0]) for probe_path in probe_list])

        # run protocol
        probe_feats = infer_images(model=model,
                                     image_list=probe_list,
                                     landmark_list_path=None,
                                     batch_size=args.batch_size,
                                     use_flip_test=args.use_flip_test,
                                     qualnet=args.qualnet,
                                     aligned=args.aligned)

        # Identification
        calc_accuracy(probe_feats, probe_labels, gallery_feats, gallery_labels, distance, args.save_dir, args.aligned, do_norm=True)
        