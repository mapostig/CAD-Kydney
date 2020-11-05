#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:05:00 2020

@author: mpostigo

"""
import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from skimage import io
from skimage.morphology import label
from skimage.measure import regionprops

class KidneyDataset(Dataset):
    
    def __init__(self, root, partition, fold, transforms=None, only_local=True):
        """
        Parameters
        ----------
        root: directory of images, masks, labels...
        partition: train test val
        fold: between 1-5. To train with different splits of the data
        transforms: the data transforms
        only_local: boolean. If only local, only local pathologies will be detected.
        Otherwise, global pathologies will be also detected.

        Returns
        -------
        None.
        """
        
        self.root = root
        self.transforms = transforms
        self.only_local = only_local
        
        # load all image files, sorting them to
        # ensure that they are aligned
        
        self.imgs_files, self.mask_files, self.annotations_files = self.load_data (partition, fold)

        if only_local:
            self.class_names = ("Kidney", "Cyst", "Pyramid", "Hydronephrosis", "Others")
        else:
            self.class_names = ("Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Bad_corticomedullary_differentiation", "Hyperechogenic_renal_cortex")
        
        """
        REMARK: BACKGROUND LABEL IS NEEDED: 0
        """
        self.class_dict = {class_name: i+1 for i, class_name in enumerate(self.class_names)}  
        self.class_dict["Background"] = 0
        # print(self.class_names)
       
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # load images and masks
        img_path = self.imgs_files[idx]
        print(img_path)
        image = io.imread(img_path)
        
        #we want the kidney mask in order to crop the kidney to transform
        mask_path = self.mask_files[idx]
        mask = sio.loadmat(mask_path)['mask']
        
        #load boxes and labels
        boxes, labels = self._get_annotation(idx)
        num_objs = boxes.shape[0]
        
        #convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if (len(boxes)==0):
            area = torch.as_tensor(0, dtype=torch.float32)
            labels = torch.as_tensor([0], dtype=torch.int64)
        else:
            area = (boxes[:, 0] + boxes[:, 2]) * (boxes[:, 1] + boxes[:, 3])
        

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["mask"] = mask
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
        target["area"] = area

        if self.transforms is not None:
            image, target = self.transforms(image, target)
                   
        return image, target

    def __len__(self):
        return len(self.imgs_files)
    
    def load_data (self, partition, fold): 
        
        """ Load the data from data_dir directory given the partition 
            [Train, Val, Test] and fold [1,2,3,4,5]
        """
        
        print("Reading data from fold %d" %(fold))
        
        indexes = sio.loadmat(os.path.join(self.root,'splits','idx'+partition+'M'+str(fold)+'.mat'))['idx'+partition][0]
        
        print("%s samples: %d" %(partition, len(indexes)))
        
        return self.get_imgs_masks_labels_annotations (indexes)

    def get_imgs_masks_labels_annotations (self, valids):
        
        
        imgs_files = [os.path.join(self.root,'images',f) for f in sorted(os.listdir(os.path.join(self.root,'images'))) 
                if (os.path.isfile(os.path.join(self.root,'images',f)) and f.endswith('.jpg'))]
        
        index = np.where(valids == 1986)
        
        valids = np.delete(valids,index)
        
        imgs_files = [imgs_files[f] for f in (valids-1)]
        
        masks_files = [os.path.join(self.root,'masks_poly',f) for f in sorted(os.listdir(os.path.join(self.root,'masks_poly'))) 
                if (os.path.isfile(os.path.join(self.root,'masks_poly',f)) and f.endswith('.mat'))]
        masks_files = [masks_files[f] for f in (valids-1)]
        
        annotations_files = [os.path.join(self.root,'annotations',f) for f in sorted(os.listdir(os.path.join(self.root,'annotations'))) 
                if (os.path.isfile(os.path.join(self.root,'annotations',f)) and f.endswith('.txt'))]
        annotations_files = [annotations_files[f] for f in (valids-1)]

        return imgs_files, masks_files, annotations_files
        
    def _get_annotation(self, idx):
        
        annotation_file = self.annotations_files[idx]
        mask_file = self.mask_files[idx]
        
        objects, string_boxes = self.find_boxes_and_objects(annotation_file, mask_file)
        
        labels = []
        boxes = []
        
        
        for obj, bbox in zip(objects, string_boxes):
        	if obj in self.class_names:
	            
	            xmin = float(bbox[0])
	            ymin = float(bbox[1])
	            w = float(bbox[2])
	            h = float(bbox[3])
	            boxes.append([xmin,ymin,xmin+w,ymin+h])
	            labels.append(self.class_dict[obj])
                
                
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def find_boxes_and_objects(self, annotation_file, mask_file):#only check for Quistes
        objects= []
        boxes = []
        
        with open(annotation_file, encoding = "ISO-8859-1") as f:
            lines = f.readlines()
        
        if (not self.only_local):
#          GLOBAL PATHOLOGIES
            str_global = lines[3].rstrip()
    #        print(a[46:])
            pat1 = int (str_global[-3])
            if (pat1==1):
                objects.append(self.class_names[5])#Bad corticomedullary differentiation
                box = self.get_mask_bbox(mask_file)
                boxes.append(box)
    #        print(global_pathologies[0],': ', str_global[-3])
            pat2 = int (str_global[-1])
            if (pat2==1):
                objects.append(self.class_names[6])#Hyperechogenic renal cortex"
                box = self.get_mask_bbox(mask_file)
                boxes.append(box)
                
        else: 
            # Label all kidneys as kidney
            objects.append(self.class_names[0])
            box = self.get_mask_bbox(mask_file)
            boxes.append(box)
            
        if(len(lines)>6):
            #LOCAL PATHOLOGIES
            Bounding_boxes = lines[6:]
            for b in Bounding_boxes: 
                l = b.rstrip()
                pat = int (l[-1])

                box_aux = l[l.index(':')+1:-1]
                
                if (pat==2 or pat==3):
                    objects.append(self.class_names[1])#Quistes
                elif (pat==4):
                     objects.append(self.class_names[2])#Piramides
                elif(pat==7):
                     objects.append(self.class_names[3])#Hidronefrosis
                else:
                     objects.append(self.class_names[4])#Otros     
                
                box = box_aux.split()
                boxes.append(box)
                
        if len(boxes)==0: #It is a healthy kidney
            objects.append(self.class_names[0])#Healthy
            box = self.get_mask_bbox(mask_file)
            boxes.append(box)

        return objects, boxes
    
    def get_mask_bbox(self, mask_file):
        
        mask = sio.loadmat(mask_file)['mask']
        lb = label (mask)
        props = regionprops(lb)
        for prop in props:
          b = [str(prop.bbox[1]), str(prop.bbox[0]),
               str(prop.bbox[3]-prop.bbox[1]), str(prop.bbox[2]-prop.bbox[0])]
        return b




