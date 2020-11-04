#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:24:20 2020

@author: mpostigo
"""
import os
import numpy as np

def write_results (ground_truth, predicted, data_dir, start_id):
    """
    This method writes the ground truth and the multilabel predictions in 
    .txt files for future evaluation metrics 
    
    Parameters
    ----------
    ground_truth : A list with the GT labels for both global pathologies
    predicted : A list with the predicted label for both global pathologies
    data_dir : The directory where the results are written
    start_id: if 0 the previous data within the dir ir deleted. Only 0 
    in the 1st fold

    Returns
    -------
    None.

    """
    
    gt_file = data_dir+"/GT.txt"
    pred_file = data_dir+"/pred.txt"
    
    if start_id == 0:
        os.remove(gt_file)
        os.remove(pred_file) 
                
    with open(gt_file, 'a') as the_file:              
        # the_file.write(ground_truth[:,0])
        np.savetxt(the_file, ground_truth, fmt = '%d')
        
    with open(pred_file, 'a') as the_file2:              
        np.savetxt(the_file2, predicted, fmt = '%.8f')
    
    print("Results saved!")