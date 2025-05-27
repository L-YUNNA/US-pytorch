from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import os
import cv2


# Define your custom augmentation class
class PartialResizer:
    def __init__(self, input_size=(500, 500), draw=False):
        #self.image = image
        self.input_size = input_size
        
    def __call__(self, image, draw=False):
        # Implement your custom augmentation logic here
        image = np.array(image)
        H, W, C = image.shape
        
        for _ in range(random.randint(1, 4)):
            resize_range = random.randint(10, 50)   # min tumor size (width) = 40 
            start_point = random.randint(0, self.input_size[0]-resize_range)
            ratio = random.uniform(0.1, 0.5)
            
            part1_img = image[:, 0:start_point, :]
            region_img = image[:, start_point:start_point+resize_range, :]
            part3_img = image[:, start_point+resize_range:W, :]
            
            # resizing
            resized_region_img = cv2.resize(region_img, dsize=(int(resize_range*ratio), H))
            reduced = region_img.shape[1] - resized_region_img.shape[1]
            new_W = W - reduced
            
            aug_img = np.concatenate([part1_img, resized_region_img, part3_img], axis=1)
            padded_image = cv2.copyMakeBorder(aug_img, 0, 0, 0, W-new_W, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # draw rectangle around resized again
            if draw:
                cv2.rectangle(padded_image, (start_point, 0), (start_point + resized_region_img.shape[1], H), (255, 0, 0), 1)
            image = padded_image
        
#         cv2.rectangle(padded_image, (start_point, 0), (start_point+resized_region_img.shape[1], H), (255, 0, 0), 1)

        return padded_image
