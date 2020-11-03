#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:30:14 2020

@author: mpostigo
María Postigo Fliquete

This is the custom dataset for binary classification (Healthy vs sick kidney)
"""
from torch.utils.data import Dataset
import DataLoad_Functions
import torch

class KidneyMaskDataset(Dataset):

    def __init__(self, data_dir, partition, fold, annotations=True, transform=None):
        """
        Args:
            data_dir: directory where the images, labels, masks are located
            partition: [Train, Val, Test]
            fold: data partition to test different combinations of the images
        """
        self.imgs, self.masks, self.labels, self.global_annotations, self.local_annotations = DataLoad_Functions.load_data (data_dir, partition, fold, annotations=True)

        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #kidney image
        image = self.imgs[idx]
        #label = 0, 1
        label = self.labels[idx]
        #kidney mask 
        mask = self.masks[idx]
        #global annotations, a 2D multilabel array for two global pathologies
        global_annotations = self.global_annotations[idx]
        #local annotations, a 4D multilabel array for four local pathologies
        local_annotations = self.local_annotations[idx]
        
        #REMARK: LOCAL AND GLOBAL ANNOTATIONS ARE UNUSED IN BINARY CLASSIFICATION

        sample = {'image': image, 'mask': mask, 'label': label, 'ga':global_annotations, 'la':local_annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample
