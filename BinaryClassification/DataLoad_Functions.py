#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:54:40 2020

@author: mpostigo
"""

import os
import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/mpostigo/Documentos/kidney/bbdd/'
fold = 1 #1-5

def get_annotations (annotation_files):
    """

    Parameters
    ----------
    annotation_files : a list with the annotation files of a given fold

    Returns
    -------
    A) The global annotations as (0-1, 0-1) according to 
    'Mala diferenciacion corticomedular', 'Cortical hiperecogenica'
    B) The local annotations as (0-1, 0-1, 0-1, 0-1) according to 'Quiste', 
    'Piramide', 'Hidrofenosis', 'Otros'
    """

    global_pathologies = ['Mala diferenciacion corticomedular', 'Cortical hiperecogenica']
#    local_pathologies = ['Litiasis','Quiste Simple', 'Quiste Complicado', 'Piramide', 'Angiomiolipoma',
#                   'Masa renal solida', 'Hidrofenosis', 'Cortical adelgazada', 'Escara Cortical']
    local_pathologies = ['Quiste', 'Piramide', 'Hidrofenosis', 'Otros']
    #hidrofenosis, quistes juntos, piramides, otros
    local_annotations = np.zeros((len(annotation_files), len(local_pathologies)))
    global_annotations = np.zeros((len(annotation_files), len(global_pathologies)))
    
    for a, row in zip(annotation_files, range(len(annotation_files))):
        
        with open(a,encoding = "ISO-8859-1") as f:
            lines = f.readlines()
        str_global = lines[3].rstrip()
        global_annotations[row,0] = str_global[-3]
        global_annotations[row,1] = str_global[-1]
        if(len(lines)>6):
            #There is a local pathology in the file
            Bounding_boxes = lines[6:]
            for b in Bounding_boxes: 
                l = b.rstrip()
                pat = int (l[-1])
                if (pat==2 or pat==3):
                    col = 0
                elif (pat==4):
                    col=1
                elif(pat==7):
                    col=2
                else:
                    col=3            
                local_annotations[row,col] = 1
        else:
            #there is not a local pathology in the file
            local_annotations[row, :] = np.zeros((1,len(local_pathologies)))
            
    print('GLOBAL PATHOLOGIES')
    print("Number of %s= %d of %d samples"%(global_pathologies[0], np.sum(global_annotations[:,0]>0), len(global_annotations)))
    print("Number of %s= %d of %d samples"%(global_pathologies[1],np.sum(global_annotations[:,1]>0), len(global_annotations)))
    
    print('LOCAL PATHOLOGIES')
    print("Number of %s= %d of %d samples"%(local_pathologies[0], np.sum(local_annotations[:,0]>0), len(local_annotations)))
    print("Number of %s= %d of %d samples"%(local_pathologies[1], np.sum(local_annotations[:,1]>0), len(local_annotations)))
    print("Number of %s= %d of %d samples"%(local_pathologies[2], np.sum(local_annotations[:,2]>0), len(local_annotations)))
    
    

    return global_annotations, local_annotations


def get_imgs_masks_labels_annotations (data_dir, valids, annotations = False):
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
    labels : list containing the labels from the given split (0,1)
    global_annotations : list containing the global annotations from the given split
    local_annotations : list containing the local annotations from the given split

    """
    
    imgs_files = [os.path.join(data_dir,'images',f) for f in sorted(os.listdir(os.path.join(data_dir,'images'))) 
            if (os.path.isfile(os.path.join(data_dir,'images',f)) and f.endswith('.jpg'))]
    imgs_files = [imgs_files[f] for f in (valids-1)]
    
    masks_files = [os.path.join(data_dir,'masks_poly',f) for f in sorted(os.listdir(os.path.join(data_dir,'masks_poly'))) 
            if (os.path.isfile(os.path.join(data_dir,'masks_poly',f)) and f.endswith('.mat'))]
    masks_files = [masks_files[f] for f in (valids-1)]
    
    labels_files = [os.path.join(data_dir,'labels',f) for f in sorted(os.listdir(os.path.join(data_dir,'labels'))) 
            if (os.path.isfile(os.path.join(data_dir,'labels',f)) and f.endswith('.mat'))]
    labels_files = [labels_files[f] for f in (valids-1)]
    
    annotations_files = [os.path.join(data_dir,'annotations',f) for f in sorted(os.listdir(os.path.join(data_dir,'annotations'))) 
            if (os.path.isfile(os.path.join(data_dir,'annotations',f)) and f.endswith('.txt'))]
    annotations_files = [annotations_files[f] for f in (valids-1)]
    
    """ Now from the given files, extract the relevant information
    """

    labels=[int(sio.loadmat(label)['label'][0][0])%2 for label in labels_files]#%2 to have 0 and 1
    images=[cv2.imread(image) for image in imgs_files]
    masks=[sio.loadmat(mask)['mask'] for mask in masks_files]# dtype('uint8')
    
    local_annotations = []
    global_annotations = []

    
    if (annotations):
        global_annotations, local_annotations = get_annotations (annotations_files)
    
    return images, masks, labels, global_annotations, local_annotations
    
    

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
    
    #Load the indexes from splits file. There are 5x3 splits files, one for each fold and partition.
    #Each split files contains a set of indexes for all the dataset
    indexes = sio.loadmat(os.path.join(data_dir,'splits','idx'+partition+'M'+str(fold)+'.mat'))['idx'+partition][0]
    
    print("%s samples: %d" %(partition, len(indexes)))

    return get_imgs_masks_labels_annotations (data_dir, indexes, annotations)
    
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
