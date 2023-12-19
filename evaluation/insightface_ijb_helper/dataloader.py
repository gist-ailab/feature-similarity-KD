import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from skimage import transform as trans


class ImageAligner:
    def __init__(self, image_size=(112, 112)):

        self.image_size = image_size
        src = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
        if self.image_size[0] == 112:
            src[:, 0] += 8.0

        self.src = src

    def align(self, img, landmark):
        # align image with pre calculated landmark

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        return img


class ListDatasetWithAligner(Dataset):
    def __init__(self, img_list, landmarks, image_size=(112,112), aligned=True, image_is_saved_with_swapped_B_and_R=False):
        super(ListDatasetWithAligner, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list

        self.aligned = aligned
        if self.aligned:
            self.landmarks = landmarks
            self.aligner = ImageAligner(image_size=image_size)
        else:
            self.landmarks = None
            self.aligner = None
            
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]

        img = cv2.imread(image_path)
        img = img[:, :, :3]

        if self.image_is_saved_with_swapped_B_and_R:
            print('check if it really should be on')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if self.aligned:
            landmark = self.landmarks[idx]
            img = self.aligner.align(img, landmark)
        else:
            img = resize_with_padding(img, target_size=112)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


def prepare_dataloader(img_list, landmarks, batch_size, num_workers=0, image_size=(112,112), aligned=True, image_is_saved_with_swapped_B_and_R=False):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True
    image_dataset = ListDatasetWithAligner(img_list, landmarks, image_size=image_size, aligned=aligned, image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader


def resize_with_padding(img, target_size=112):
    old_size = img.shape[:2] # old_size is in (height, width) format

    # Compute the scaling factor
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # Resize the image
    img = cv2.resize(img, (new_size[1], new_size[0]), cv2.INTER_LINEAR)

    # Compute deltas to add to make the image square
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]

    # Calculate padding
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Add padding to make the image square
    color = [0, 0, 0] # Padding color (black)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img