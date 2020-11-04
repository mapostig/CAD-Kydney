#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:52:08 2020

@author: mpostigo

Custom dataset for multi-label global image classification
Here the labels are binary vectors
"""

from torch.utils.data import Dataset
import torch
from DataLoadFunctions import *

class KidneyMaskDataset(Dataset):

    def __init__(self, data_dir, partition, fold, annotations=True, transform=None):
        """
        Args:
            data_dir: directory where the images are located
            partition: [Train, Val, Test]
            fold: data partition to test different combinations of the images
        """
        self.imgs, self.masks, self.labels = load_data (data_dir, partition, fold, annotations=True)

        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgs[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        sample = {'image': image, 'mask': mask, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample