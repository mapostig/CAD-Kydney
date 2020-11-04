#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:53:14 2020

@author: mpostigo
"""

import os
import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Not global pathology
Mala diferenciacion corticomedular
Cortical hiperecogenica
"""

def get_annotations (annotation_files):
    """
    Parameters
    ----------
    annotation_files : annotation files list from a given fold

    Returns
    -------
    annotations : returns a list containing the 1x7 arrays with the annotation
    of each image. N images --> Nx7 binary matrix

    """
   
    global_pathologies = ['Mala diferenciacion corticomedular', 'Cortical hiperecogenica']
    local_pathologies = ['Quiste', 'Piramide', 'Hidrofenosis', 'Otros']
    annotations = np.zeros((len(annotation_files), 1+len(global_pathologies)+len(local_pathologies)))
    
    
    for a, row in zip(annotation_files, range(len(annotation_files))):
        
        with open(a,encoding = "ISO-8859-1") as f:
            lines = f.readlines()
        str_global = lines[3].rstrip()
        annotations[row,5] = int(str_global[-3])#Bad corticomedullar differenciation
        annotations[row,6] = int(str_global[-1])#Hyperecogenic cortex
        
        if(len(lines)>6):
            Bounding_boxes = lines[6:]
            for b in Bounding_boxes: 
                l = b.rstrip()
                pat = int (l[-1])
                if (pat==2 or pat==3):
                    col = 1 #Cyst
                elif (pat==4):
                    col=2 #Pyramid
                elif(pat==7):
                    col=3 #Hydroneprhosis
                else:
                    col=4 #Others           
                annotations[row,col] = 1
        if max(annotations[row,:])==0: #it is a healthy kidney
            annotations[row,0] = 1

    return annotations


def get_imgs_masks_labels_annotations (data_dir, valids, annotations = True):
    """
    Parameters
    ----------
    data_dir : directory with the images, labels, masks and annotations
    valids : The indexes corresponding to the given split (according to fold and
    partition)
    annotations : boolean to load the annotations or not
    
    Returns
    -------
    images : list containing the images from the given split
    masks : list containing the masks from the given split
    labels : list containing the labels from the given split 
    The labels are binary vectors according to the present/absense of a given class
    class_names = ["Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Poor Corticomedular Differenciation", "Hyperechogenic cortex"]
    label example: [0,1,1,1,0,0,1]
    """
    index = np.where(valids == 1986)
        
    valids = np.delete(valids,index)

    imgs_files = [os.path.join(data_dir,'images',f) for f in sorted(os.listdir(os.path.join(data_dir,'images'))) 
            if (os.path.isfile(os.path.join(data_dir,'images',f)) and f.endswith('.jpg'))]
    imgs_files = [imgs_files[f] for f in (valids-1)]
    
    masks_files = [os.path.join(data_dir,'masks_poly',f) for f in sorted(os.listdir(os.path.join(data_dir,'masks_poly'))) 
            if (os.path.isfile(os.path.join(data_dir,'masks_poly',f)) and f.endswith('.mat'))]
    masks_files = [masks_files[f] for f in (valids-1)]
    
    annotations_files = [os.path.join(data_dir,'annotations',f) for f in sorted(os.listdir(os.path.join(data_dir,'annotations'))) 
            if (os.path.isfile(os.path.join(data_dir,'annotations',f)) and f.endswith('.txt'))]
    annotations_files = [annotations_files[f] for f in (valids-1)]
    
    """ Now from the given files, extract the relevant information
    """
    images=[cv2.imread(image) for image in imgs_files]
    masks=[sio.loadmat(mask)['mask'] for mask in masks_files]# dtype('uint8')

    all_annotations=[]
    if (annotations):
        all_annotations = get_annotations (annotations_files)
    
    labels = all_annotations
    
    return images, masks, labels
    
    

def load_data (data_dir, partition, fold, annotations=True): 
    
    """
    Parameters
    ----------
    data_dir : directory where the images, labels, masks, annotations and splits 
    are located
    partition : [Train, Test, Val]
    fold : from 1-5 different folds for training
    annotations : boolean to load the global and local annotations or not
    Returns
    -------
    A list with all the images, masks, labels and annotations (if annotations=True)
    """
    print("Reading data from fold %d" %(fold))
    
    indexes = sio.loadmat(os.path.join(data_dir,'splits','idx'+partition+'M'+str(fold)+'.mat'))['idx'+partition][0]
    
    print("%s samples: %d" %(partition, len(indexes)))

    return get_imgs_masks_labels_annotations (data_dir, indexes, annotations)
    
def show_masks(image, mask):
    """Show image with mask"""
    plt.imshow(image)
    plt.imshow(mask, cmap='gray', alpha=0.5)

def load_mean_std (data_dir, fold):
    """
    Parameters
    ----------
    data_dir : Directory containing the mean and std per training fold
    fold : from 1-5
    Returns
    -------
    mean : mean of the training set
    std : std of the training set
    """
    
    mean=sio.loadmat(os.path.join(data_dir,'splits','rgbM'+str(fold)+'.mat'))['rgb_mean']
    std=sio.loadmat(os.path.join(data_dir,'splits','stdM'+str(fold)+'.mat'))['rgb_cov']
    std=np.sqrt(np.diag(std))
    print("Mean", mean)
    print("std", std)
    return mean, std