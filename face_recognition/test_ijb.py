import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from backbone.iresnet import iresnet18, iresnet50
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
from evaluation.insightface_ijb_helper.dataloader import prepare_dataloader
from evaluation.insightface_ijb_helper import eval_helper_identification
from evaluation.insightface_ijb_helper import eval_helper as eval_helper_verification

import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict


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


def infer_images(model, img_root, landmark_list_path, batch_size, use_flip_test, qualnet=False, aligned=True):
    img_list = open(landmark_list_path)
    # img_aligner = ImageAligner(image_size=(112, 112))

    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_paths = []
    landmarks = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(img_root, name_lmk_score[0])
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_paths.append(img_path)
        landmarks.append(lmk)
        faceness_scores.append(name_lmk_score[-1])

    print('total images : {}'.format(len(img_paths)))
    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=0, image_size=(112,112), aligned=aligned)

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
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    assert len(features) == len(img_paths)
    return img_feats, faceness_scores


def identification(data_root, dataset_name, img_input_feats, save_dir, aligned):
    # Step1: Load Meta Data
    meta_dir = os.path.join(data_root, dataset_name, 'meta')
    if dataset_name == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (dataset_name.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (dataset_name.lower())

    gallery_s1_templates, gallery_s1_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % dataset_name.lower()
    total_templates, total_medias = eval_helper_identification.read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)

    # # Step2: Get gallery Features
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates, gallery_subject_ids)
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)

    # # step 4 get probe features
    probe_mixed_record = "%s_1N_probe_mixed.csv" % dataset_name.lower()
    probe_mixed_templates, probe_mixed_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = eval_helper_identification.gen_mask(probe_ids, gallery_ids)
    identification_result = eval_helper_identification.evaluation(probe_feats, gallery_feats, mask)

    if aligned:
        pd.DataFrame(identification_result, index=['identification']).to_csv(os.path.join(save_dir, "identification_aligned_result.csv"))
    else:
        pd.DataFrame(identification_result, index=['identification']).to_csv(os.path.join(save_dir, "identification_not_result.csv"))

    
def verification(data_root, dataset_name, img_input_feats, save_dir, prefix, aligned):
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(data_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    p1, p2, label = eval_helper_verification.read_template_pair_list(
        os.path.join(data_root, '%s/meta' % dataset_name,
                    '%s_template_pair_label.txt' % dataset_name.lower()))

    template_norm_feats, unique_templates = eval_helper_verification.image2template_feature(img_input_feats, templates, medias)
    score = eval_helper_verification.verification(template_norm_feats, unique_templates, p1, p2)

    # # Step 5: Get ROC Curves and TPR@FPR Table
    score_save_file = os.path.join(save_dir, "verification_score_%s.npy" %prefix)
    np.save(score_save_file, score)
    result_files = [score_save_file]

    save_path = os.path.join(save_dir, prefix + '.pkl')
    eval_helper_verification.write_result(result_files, save_path, aligned, dataset_name, label)
    os.remove(score_save_file)


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
    parser.add_argument('--data_dir', default='/home/jovyan/SSDb/sung/dataset/face_dset/ijb')
    parser.add_argument('--data_name', default='IJBB', type=str, help='dataset_name, set to IJBC or IJBB')
    parser.add_argument('--gpus', default='3', type=str)
    parser.add_argument('--batch_size', default=512, type=int, help='')
    parser.add_argument('--mode', type=str, default='ir', help='attention type')
    parser.add_argument('--backbone', type=str, default='iresnet50')
    parser.add_argument('--pooling', type=str, default='E') #
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/student-casia/iresnet50-E-IR-ArcFace/resol1-random/F_SKD_CROSS_BN-P{20.0,4.0}-M{0.0}/seed{5}', help='scale size')
    parser.add_argument('--save_dir', type=str, default='result/', help='scale size')
    parser.add_argument('--prefix', type=str, default='aa', help='scale size')

    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--qualnet', type=str2bool, default='False')
    parser.add_argument('--aligned', type=str2bool, default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Dataset
    data_name = args.data_name
    assert data_name in ['IJBC', 'IJBB']
    print('data name: ', data_name)
    print('use_flip_test', args.use_flip_test)
    print('aligned', args.aligned)

    # args.save_dir = os.path.join(os.path.dirname(args.checkpoint_path), '%s_result' %args.data_name.lower())
    os.makedirs(args.save_dir, exist_ok=True)

    model = load_model(args)

    # get features and fuse
    img_root = os.path.join(args.data_dir, '%s/loose_crop' % data_name)
    landmark_list_path = os.path.join(args.data_dir, '%s/meta/%s_name_5pts_score.txt' % (data_name, data_name.lower()))
    img_input_feats, faceness_scores = infer_images(model=model,
                                                    img_root=img_root,
                                                    landmark_list_path=landmark_list_path,
                                                    batch_size=args.batch_size,
                                                    use_flip_test=args.use_flip_test,
                                                    qualnet=args.qualnet,
                                                    aligned=args.aligned)
    print('Feature Shape: ({} , {}) .'.format(img_input_feats.shape[0], img_input_feats.shape[1]))

    # run protocol
    # identification(args.data_dir, data_name, img_input_feats, save_dir=args.save_dir, aligned=args.aligned)
    verification(args.data_dir, data_name, img_input_feats, save_dir=args.save_dir, prefix=args.prefix, aligned=args.aligned)