#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:34:57 2020

@author: mpostigo
"""
import numpy as np
import scipy.io as sio
import os, shutil
import re
import cv2
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import img_as_ubyte, img_as_float64
import os
import glob


def create_detection_results_files (images, targets, outputs, start_id, only_local= True):
    """
    Create a set of files with the predicted bounding boxes and their corresponding labels
    in order to obtain the mAP executing  https://github.com/Cartucho/mAP
    Args: 
        images: all the test images
        targets: all the test targets (ground truth)
        outputs: all the predictions (labels, boxes and scores)
        start_id: to rename the images from each fold
    """
    
    if only_local:
        class_names = ("Kidney", "Cyst", "Pyramid", "Hydronephrosis", "Others")
    else:
        class_names = ("Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
        "PCD", "HC")  
        
    label_map = {class_name: i+1 for i, class_name in enumerate(class_names)}  
    label_map["Background"] = 0
    
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    if start_id == 0:
        
        mydir = '/home/mpostigo/Documentos/bbdd/mAP-master/input/ground-truth'
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f))
                
        mydir = '/home/mpostigo/Documentos/bbdd/mAP-master/input/detection-results'
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f))   
            
        mydir = '/home/mpostigo/Documentos/bbdd/mAP-master/input/images-optional'
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".jpg") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f)) 
        
    output_path = '/home/mpostigo/Documentos/bbdd/mAP-master/input/ground-truth/'
    output_path2 = '/home/mpostigo/Documentos/bbdd/mAP-master/input/detection-results/'
       
    for i, (image, target, output) in enumerate(zip(images, targets, outputs)):
        
        batch_size = len(target)
        
        for b in range(batch_size):
            GT = []
            D = []
            idx = target[b]['image_id'].detach().cpu().numpy()
            idx = idx+start_id
            #CREATE GT files
            filename1 = output_path+'image_'+str(idx)+'.txt'
            labels = target[b]['labels']
            boxes = target[b]['boxes']
            
            #CREATE DETECTION FILES
            filename2 = output_path2+'image_'+str(idx)+'.txt'
            pred_labels = output[b]['labels']
            scores = output[b]['scores']
            pred_boxes = output[b]['boxes']
            
            scores = scores.numpy()
            pred_labels = pred_labels.numpy()
            pred_boxes = pred_boxes.numpy()
            
            #Clean healthy score
            healthy_indexes = np.where(pred_labels==1)
            sick_indexes = np.where(pred_labels!=1)
            if (np.asarray(sick_indexes).size !=0) :
                if(np.amax(scores[sick_indexes])>=0.7):
                    scores = np.delete(scores,healthy_indexes) 
                    pred_labels = np.delete(pred_labels,healthy_indexes)
                    pred_boxes = np.delete(pred_boxes,healthy_indexes, axis=0)
                    
            # #Clean sick scores        
            healthy_indexes = np.where(pred_labels==1)   
            sick_indexes = np.where(pred_labels!=1)
            if np.asarray(healthy_indexes).size != 0 :        
                if(np.amax(scores[healthy_indexes])>=0.9):
                    scores = np.delete(scores,sick_indexes) 
                    pred_labels = np.delete(pred_labels,sick_indexes)
                    pred_boxes = np.delete(pred_boxes,sick_indexes, axis=0)
                    
                elif(np.amax(scores[healthy_indexes])<=0.7):
                    scores = np.delete(scores,healthy_indexes) 
                    pred_labels = np.delete(pred_labels,healthy_indexes)
                    pred_boxes = np.delete(pred_boxes,healthy_indexes, axis=0)
                     
            #Clean low scores local
            local_indexes = np.where((pred_labels==2) | (pred_labels==3) | (pred_labels==4) | (pred_labels==5)| (pred_labels==7))
            low_scores_idx = np.where(scores<0.5)
            low_scores_idx_local = np.intersect1d(local_indexes, low_scores_idx)
            
            if np.asarray(low_scores_idx_local).size!=0:
                scores = np.delete(scores,low_scores_idx_local)
                pred_labels = np.delete(pred_labels,low_scores_idx_local)
                pred_boxes = np.delete(pred_boxes,low_scores_idx_local, axis=0)
                    
            #clean low scores global
            global_indexes = np.where((pred_labels==6))
            low_scores_idx = np.where(scores<0.2)
            low_scores_idx_global = np.intersect1d(global_indexes, low_scores_idx)
            
            if np.asarray(low_scores_idx_global).size !=0:
                scores = np.delete(scores,low_scores_idx_global)
                pred_labels = np.delete(pred_labels,low_scores_idx_global)
                pred_boxes = np.delete(pred_boxes,low_scores_idx_global, axis=0)
                
            #CREATE IMAGES FILES
            img = image[b].cpu()
            img = img.numpy().transpose((1, 2, 0))
#            plt.imshow(img)
            # im_path = '/home/mpostigo/Documentos/bbdd/mAP-master/input/images-optional/image_'+str(idx)+'.jpg'
            # skio.imsave(im_path, img_as_float64(img/np.amax(np.absolute(img))))

            for l,b in zip (labels.numpy(), boxes.numpy()):
                
                str_label = rev_label_map.get(l).replace(' ','_')
                str_box = str(b).replace('[','').replace(']','').replace(',','')
                file_line = str_label+' '+str_box+'\n'
                GT.append(' '.join(file_line.split()))
            
            for pl, s, pb in zip (pred_labels, scores, pred_boxes):
                
                str_plabel = rev_label_map.get(pl).replace(' ','_')
                str_score = str(s)
                str_pbox = str(pb).replace('[','').replace(']','').replace(',','')
                
                file_line2 = str_plabel+' '+str_score+' '+str_pbox+'\n'
                
                D.append(' '.join(file_line2.split()))
            
            with open(filename1, 'w') as the_file:
                    the_file.write('\n'.join(GT))
                    
            with open(filename2, 'w') as the_file2:
                the_file2.write('\n'.join(D))
                
            the_file.close()
            the_file2.close()
            
def write_results_multiclass (ground_truth, predicted, data_dir, approach):
    """
    Parameters
    ----------
    ground_truth : A list with the GT labels for both global pathologies
    predicted : A list with the predicted label for both global pathologies
    data_dir : The directory where the results are written
    start_id: if 0 the previous data within the dir ir deleted. Only 0 
    in the 1st fold
    approach: [max, area, mean, log, sum]

    Returns
    -------
    None.

    """
    
    gt_file = data_dir+"/GT_"+approach+".txt"
    pred_file = data_dir+"/pred_"+approach+".txt"
    
    
    os.remove(gt_file)
    os.remove(pred_file) 
                
    with open(gt_file, 'a') as the_file:              
        # the_file.write(ground_truth[:,0])
        np.savetxt(the_file, ground_truth, fmt = '%d')
        
    with open(pred_file, 'a') as the_file2:              
        np.savetxt(the_file2, predicted, fmt = '%.8f')
    
    print("Results saved!")
    

def save_outputs (targets, outputs, start_id, only_local=False):
    """
    Create a set of files with the predicted bounding boxes and their corresponding labels
    Args: 
        targets: all the test targets (ground truth)
        outputs: all the predictions (labels, boxes and scores)
    """
    
    if only_local:
        class_names = ("Kidney", "Cyst", "Pyramid", "Hydronephrosis", "Others")
    else:
        class_names = ("Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
        "PCD", "HC")  
        
    label_map = {class_name: i+1 for i, class_name in enumerate(class_names)}  
    label_map["Background"] = 0
    
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    if start_id == 0:
  
        mydir = '/home/mpostigo/Documentos/bbdd/output_results/targets'
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f))   
            
        mydir = '/home/mpostigo/Documentos/bbdd/output_results/outputs'
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f)) 
        
    output_path = '/home/mpostigo/Documentos/bbdd/output_results/'
    
    for i, (target, output) in enumerate(zip(targets, outputs)):
        
        batch_size = len(target)
        
        for b in range(batch_size):
            GT=[]
            D=[]
            
            idx = target[b]['image_id'].detach().cpu().numpy()
            idx = idx+start_id
            
            #CREATE GT files
#            filename = output_path+filenames[idx]+'.txt'
            filename1 = output_path +'targets/'+'image_'+str(idx)+'.txt'
            labels = target[b]['labels']
    #        scores = target['scores']
            boxes = target[b]['boxes']
            
            #CREATE DETECTION FILES
            filename2 = output_path+'outputs/'+'image_'+str(idx)+'.txt'
            pred_labels = output[b]['labels']
            scores = output[b]['scores']
            pred_boxes = output[b]['boxes']
            
            scores = scores.numpy()
            pred_labels = pred_labels.numpy()
            pred_boxes = pred_boxes.numpy()
            
            for l,b in zip (labels.numpy(), boxes.numpy()):
                
                str_label = str(l)
                str_box = str(b).replace('[','').replace(']','').replace(',','')
                file_line = str_label+' '+str_box+'\n'
                
                GT.append(' '.join(file_line.split()))
            
            for pl, s, pb in zip (pred_labels, scores, pred_boxes):
                
                # if (s>0.2):#only write imgs with high confidence level
                str_plabel = str(pl)
                str_score = str(s)
                str_pbox = str(pb).replace('[','').replace(']','').replace(',','') 
                
                file_line2 = str_plabel+' '+str_score+' '+str_pbox+'\n'
                
                D.append(' '.join(file_line2.split()))
            
            with open(filename1, 'w') as the_file:
#                    file_line.replace('   ', ' ')
                the_file.write('\n'.join(GT))
                    
            with open(filename2, 'w') as the_file2:
#                    file_line2.replace('   ', ' ')
                the_file2.write('\n'.join(D))
                
            the_file.close()
            the_file2.close()
            

def read_output_files (data_dir):
    
    targets = []
    outputs = []
    images = []
    target_dir = data_dir+'targets/'
    output_dir = data_dir+'outputs/'
    images_dir = data_dir+'images/'
    n_idx = len(os.listdir(images_dir))
    
    target_filenames = [target_dir+'image_'+str(idx)+'.txt' for idx in range(n_idx)]
    output_filenames = [output_dir+'image_'+str(idx)+'.txt' for idx in range(n_idx)]
    images_filenames = [images_dir+'image_'+str(idx)+'.jpg' for idx in range(n_idx)]
    
    for filename in target_filenames:
        with open((filename), 'r') as f:
           target = dict()
           lines = f.readlines()
           nplines = np.asarray([np.fromstring(line, dtype=float, sep=' ') for line in lines])
           labels = nplines [:,0]
           boxes = nplines[:,1:]
           target['boxes'] = np.array(boxes, dtype=np.float32)
           target['labels'] = np.array(labels, dtype=np.int64)
           targets.append(target)
           f.close()
           
    for filename in output_filenames:
            with open((filename), 'r') as f:
               output = dict()
               lines = f.readlines()
               nplines = np.asarray([np.fromstring(line, dtype=float, sep=' ') for line in lines])
               labels = nplines [:,0]
               scores = nplines [:,1]
               boxes = nplines[:,2:]
               output['boxes'] = np.array(boxes, dtype=np.float32)
               output['labels'] = np.array(labels, dtype=np.int64)
               output['scores'] = np.array(scores, dtype=np.float32)
               outputs.append(output)
               f.close()
               
    for filename in images_filenames:
           img = cv2.imread(filename)
           if img is not None:
                images.append(img)
           
    return images, targets, outputs
             
  