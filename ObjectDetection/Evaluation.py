#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:22:12 2020

@author: mpostigo
"""

import torch
import numpy as np
from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter
import torch.nn.functional as F
import time
#from utils import MetricLogger

def calc_iou(gt_bbox, pred_bbox, gt_label, pred_label):
    """
    Calculate the IoU between 2 bounding boxes
    Args:
        gt_box: corresponds to the ground truth bounding box
        pred_box:is the predicted bounding box
        gt_label: corresponds to the ground truth label
        pred_label:is the predicted label
    """
    if (pred_label != gt_label):
        return 0.0
    
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
    if(x_bottomright_gt< x_topleft_p)or(y_bottomright_gt< y_topleft_p)or(x_topleft_gt> x_bottomright_p)or(y_topleft_gt> y_bottomright_p):
        return 0.0    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

import copy

@torch.no_grad()
def evaluate(model, data_loader, device, th, phase):
    """
    Evaludates the trained model over the validation/test dataloader
    Args: 
        model: the trained model
        data_loader: validation/test dataloader
        device: either cpu or GPU
        th: the IoU threshold to consider a match
        phase: validation/test (string)
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
#    cpu_device = torch.device("cpu")

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = phase
    gt = 0
    ious = 0
    n_images = 0

    test_images = []
    test_targets = []
    test_outputs = []
    best_model_wts = copy.deepcopy(model.state_dict())

    for i, (images, targets) in tqdm(enumerate(data_loader)):

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        batch_size = len(outputs)
        

        for b in range(batch_size):
            n_images += 1
            outputs[b]['scores'] = outputs[b]['scores'].detach().cpu()
            outputs[b]['boxes'] = outputs[b]['boxes'].detach().cpu()
            outputs[b]['labels'] = outputs[b]['labels'].detach().cpu()
            
            targets[b]['boxes'] = targets[b]['boxes'].detach().cpu()
            targets[b]['labels'] = targets[b]['labels'].detach().cpu()
#
            for gt_box, gt_label in zip(targets[b]['boxes'],  targets[b]['labels']):
    
              gt += 1
              ious_list = []
    
              for pred_box, pred_label in zip(outputs[b]['boxes'], outputs[b]['labels']):
                  
                  ious_list.append(calc_iou(gt_box, pred_box, gt_label, pred_label))
    
              if len(ious_list)>0 and max(ious_list)>th:
                ious += 1
            
        test_outputs.append(outputs)
        test_images.append(images)
        test_targets.append(targets)    
            
        model_time = time.time() - model_time 
        
    evaluator_time = time.time()
    evaluator_time = time.time() - evaluator_time
    metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.set_num_threads(n_threads)

    acc = ious / gt
    print("GT boxes: ", gt)
    print("ious matched: ", ious)
    print("Analyzed images: ", n_images)

##     Deep copy of the best model
#    if phase == 'val' and acc > best_acc:
#        best_acc = acc  
#        best_model_wts = copy.deepcopy(model.state_dict())
#        # load best model weights
#        model.load_state_dict(best_model_wts)

    return acc, test_images, test_targets, test_outputs