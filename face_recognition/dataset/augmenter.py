import numpy as np
import cv2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms
import random


class Augmenter():
    def __init__(self, photometric_augmentation_prob, low_res_augmentation_prob, size_type):
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

        self.size_type = size_type

        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    def augment(self, sample):
        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)
        return sample


    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape

        if self.size_type == 'range':
            side_ratio = np.random.uniform(0.2, 1.0)
            small_side = int(side_ratio * img_shape[0])
        elif self.size_type == 'fix':
            small_side = random.sample([14, 28, 56, 112], k=1)[0]
        else:
            raise('Error!')

        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)
        return aug_img, side_ratio


    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample