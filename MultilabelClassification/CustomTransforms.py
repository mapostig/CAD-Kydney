#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:52:39 2020

@author: mpostigo
"""

import random
import numpy as np
import cv2
import torch
from torchvision.transforms import ColorJitter
from skimage import transform


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

class applyMaskTransform (object):
    """
        kernel_size: the margin around the kidney
        dim: the dimensions of the new image (224x224)
        img: kidney image
        mask: the mask
        This method will return a cropped image focused on the kidney
    """
    def __init__(self, kernel_size, dim):
        self.kernel_size = kernel_size
        self.dim = dim

    def __call__(self, sample):
        
        img = sample['image']
        label = sample['label']
        mask = sample['mask']
        kernel=np.ones((self.kernel_size,self.kernel_size), np.uint8)  
        #dilate the mask to avoid loosing info near the kidney
        dilated_mask = cv2.dilate(mask, kernel, iterations=1) 
        x,y,w,h = cv2.boundingRect(dilated_mask)
        #crop the image
        crop_img = img[y:y+h, x:x+w,:]
        res = cv2.resize(crop_img, dsize=(self.dim, self.dim), interpolation=cv2.INTER_CUBIC)
        crop_mask = mask[y:y+h, x:x+w]
        res_mask = cv2.resize(crop_mask, dsize=(self.dim, self.dim), interpolation=cv2.INTER_CUBIC)
        
        
        return {'image': res, 'mask': res_mask, 'label': label}
     
class Rescale(object):
    """Re-scale image to a predefined size.

    Args:
        output_size (tuple or int): The desired size. If it is a tuple, output is the output_size. 
        If it is an int, the smallest dimension will be the output_size
            a we will keep fixed the original aspect ratio.
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        image, mask, label = sample['image'], sample['mask'],sample['label']

        res = cv2.resize(image, dsize=(self.dim, self.dim), interpolation=cv2.INTER_CUBIC)

        res_mask = cv2.resize(mask, dsize=(self.dim, self.dim), interpolation=cv2.INTER_CUBIC)

        return {'image': res, 'mask': res_mask, 'label' : label}
    
class MyRotationTransform (object):
    """Rotate by 
        angle: 56*random.random()-28 """

    def __init__(self):
        self.angle = 56*random.random()-28

    def __call__(self, sample):
        
        img = sample['image']
        label = sample['label']
        mask = sample['mask']
        rot_img = rotate_image(img, self.angle)
        rot_mask = rotate_image(mask, self.angle)
        
        return {'image': rot_img, 'mask': rot_mask, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask, label = sample['image'], sample['mask'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(np.flip(image,axis=0).copy()) ,
                'mask': torch.from_numpy(np.flip(mask,axis=0).copy()),
                'label': torch.from_numpy(np.asarray(label))}
    
class MyHorizontalFlipTransform(object):
    
    """Horizontally flip the given numpy ndarray randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample (dict): contains the image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        img = sample['image']
        mask = sample['mask']
        if random.random() < self.p:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        return {'image': img, 'mask': mask, 'label': sample['label']}
    

class Normalize(object):
    """
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        
        img = sample['image'].type(torch.FloatTensor) 
        dtype = img.dtype
        mask = sample['mask']
        mean = torch.as_tensor(self.mean, dtype=dtype, device=img.device)
        std = torch.as_tensor(self.std[:,np.newaxis], dtype=dtype, device=img.device)
        img = (img - mean[:, None]) / std[:, None]
#        img.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        
        return {'image': img, 'mask': mask, 'label': sample['label']}
    
    
class RandomIncreaseBrightness (object):
    
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        
        img = sample ['image']
        if random.random() < self.p:
            alpha = random.choice(np.linspace(0.5,2,16))
            beta = random.choice(np.linspace(0,50,51))
            img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype),0, beta)
            
        return {'image': img, 'mask': sample ['mask'], 'label': sample['label']}