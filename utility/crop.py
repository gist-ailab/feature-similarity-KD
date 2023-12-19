from glob import glob
import os
# from facenet_pytorch import MTCNN
from PIL import Image
'''
pip install facenet-pytorch
'''
from tqdm import tqdm
import subprocess

if __name__=='__main__':
    ######################################################################################################
    # data_dir = '/home/jovyan/SSDb/sung/dataset/face_dset/scface/'
    # probe_list = glob(os.path.join(data_dir, 'surveillance_cameras_all/*_cam1_*.jpg')) + glob(os.path.join(data_dir, 'surveillance_cameras_all/*_cam2_*.jpg')) +\
    #             glob(os.path.join(data_dir, 'surveillance_cameras_all/*_cam3_*.jpg')) + glob(os.path.join(data_dir, 'surveillance_cameras_all/*_cam4_*.jpg')) + glob(os.path.join(data_dir, 'surveillance_cameras_all/*_cam5_*.jpg'))
    
    # gallery_list = glob(os.path.join(data_dir, 'mugshot_frontal_cropped_all/*.JPG'))

    # os.makedirs('examples/gallery', exist_ok=True)
    # os.makedirs('examples/probe', exist_ok=True)

    # for image_path in tqdm(gallery_list):
    #     img = Image.open(image_path)
    #     mtcnn = MTCNN(image_size=112, margin=4)
    #     out = mtcnn(img, save_path='examples/gallery/%s.jpg' %os.path.basename(image_path))
    
    # for image_path in tqdm(probe_list):
    #     img = Image.open(image_path)
    #     mtcnn = MTCNN(image_size=112, margin=4)
    #     out = mtcnn(img, save_path='examples/probe/%s.jpg' %os.path.basename(image_path))

    ######################################################################################################
    # # Missing File Selection
    # crop_gallery_list = glob('examples/gallery/*.jpg')
    # crop_probe_list = glob('examples/probe/*.jpg')

    # crop_probe_list = [os.path.basename(path) for path in crop_probe_list]
    # for path in probe_list:
    #     path = os.path.basename(path)
    #     if (path + '.jpg') not in crop_probe_list:
    #         print(path)

    ######################################################################################################
    # with open('examples/update_list.txt', 'r') as f:
    #     update_list = f.readlines()
    #     update_list = [path.strip() for path in update_list]

    # for image_path in tqdm(gallery_list):
    #     filename = os.path.basename(image_path)
    #     if filename in update_list:
    #         print(filename)
    #         new_path = 'examples/update_list/%s' %(filename)
    #         subprocess.call('cp -r %s %s' %(image_path, new_path), shell=True)



    ########################################################################################################
    base = '/home/jovyan/SSDb/sung/src/feature-similarity-KD/utility/examples'
    
    # update_list = glob(os.path.join(base, 'crop_images', '*.jpg'))
    # update_name_list = [os.path.basename(name).rstrip('.jpg') for name in update_list]

    # output_list = []
    # old_list = glob(os.path.join(base, 'probe_old', '*.jpg'))
    
    # for path in tqdm(old_list):
    #     prefix = os.path.basename(path).rstrip('.jpg')
    #     if prefix in update_name_list:
    #         print(prefix)
    #         continue
    #     else:
    #         output_list.append(path)

    # final_list = output_list + update_list

    # # Save
    # save_dir = os.path.join(base, 'probe')
    # os.makedirs(save_dir, exist_ok=True)

    # for old_path in final_list:
    #     new_path = os.path.join(save_dir, os.path.basename(old_path).rstrip('.jpg') + '.jpg')
    #     subprocess.call('cp -r %s %s' %(old_path, new_path), shell=True)


    ###################################################################################################
    base = '/home/jovyan/SSDb/sung/src/feature-similarity-KD/utility/examples'
    
    gallery_list = glob(os.path.join(base, 'gallery', '*.jpg'))
    probe_list = glob(os.path.join(base, 'probe', '*.jpg'))
    print(len(gallery_list), len(probe_list))
