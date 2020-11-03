#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 20 12:25:14 2020

@author: mpostigo

Script to try different models over different folds
Do the results analysis of the output obatined by a given model. 
Evaluates hoy the binary classifications works over the multilabel 
pathologies
"""
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def perf_measure(y_actual, y_hat):
    """
    Parameters
    ----------
    y_actual : ground truth
    y_hat : predictions

    Returns
    -------
    TP : true positives
    FP : false positives
    TN : true negatives
    FN : false negatives
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


def find_nearest(array, value):
    """
    To calculate the Specificity 95 from the ROC curve
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

#1. get the annotations

data_dir = '/home/mpostigo/Documentos/kidney/bbdd/'#This file contains the splits
model_path = '/home/mpostigo/Documentos/Models/resnet50/' #this file contains the results of the models
gt_path = '/home/mpostigo/Documentos/Models/GroundTruth/' #this file contains the ground truths

partition = 'Test'
folds = [1,2,3,4,5]

pathologies = ['Mala_diferenciacion','Hipercogenica','Quiste','Piramide',
               'Hidrofenosis', 'Otros']

len_data = 1991 #the number of images
class_true = np.zeros(len_data)
class_pred = np.zeros((len_data,2))
class_true_local = np.zeros((len_data,4))    
class_true_global = np.zeros((len_data,2))
start=0

for fold in folds:
    
    print('Executing Fold %d'%(fold))
     
    scores = sio.loadmat(model_path+'resutls_eb1_Fold'+str(fold)+'.mat')['scores']
    
    ground_truth = sio.loadmat(gt_path+'groundTruth_Fold'+str(fold)+'.mat')
    
    labels = ground_truth['labels'][0]

    end = start+ len(labels)
    
    class_true[start:end]=labels
    class_pred[start:end]=scores
    
    #locales 'Quiste', 'Piramide', 'Hidrofenosis','Otros'
    class_true_local[start:end,0] = ground_truth ['Quiste']
    class_true_local[start:end,1] = ground_truth ['Piramide']
    class_true_local[start:end,2] = ground_truth ['Hidrofenosis']
    class_true_local[start:end,3] = ground_truth ['Otros']
    
    #gobales 'Hipercogenica', 'Mala_diferenciacion'
    class_true_global[start:end,0] = ground_truth ['Hipercogenica']
    class_true_global[start:end,1] = ground_truth ['Mala_diferenciacion']
    
    start= start + len(labels)

average_precision = metrics.average_precision_score(class_true, class_pred[:,1])
roc_auc = metrics.roc_auc_score(class_true, class_pred[:,1])

red = 'resnet50'
print("SANO VS PATOLOGICO:",red)
print('--ROC SCORE %.4f'%(roc_auc))
print('--AVG score %.4f'%(average_precision))
labels = np.where(class_pred[:,1]>=0.5, 1, 0)
TP, FP, TN, FN = perf_measure(class_true, labels)
se = TP/(TP+FN)
sp = TN/ (TN+FP)
print('TP, FP, TN, FN', TP, FP, TN, FN)
print('--Sensitivity: %.4f'%(se))
print('--Specificity: %.4f'%(sp))

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(class_true, class_pred[:,1])
se95, idx = find_nearest(tpr, 0.95)
sp95 = 1 - fpr[idx]
print('--Specificity 95: %.4f'%(sp95))

local_pato = ['Quiste', 'Piramide', 'Hidrofenosis','Otros']
print('LOCAL PATHOLOGIES')
for i in range(4):
    
    labels_local = class_true_local[:,i]
    #get the indexes with the actual pathology and the healthy
    indexes = np.where((labels_local==1) | (class_true==0))
    
    labels_local = labels_local[indexes]
    class_pred_local = class_pred [indexes]
    average_precision = metrics.average_precision_score(labels_local, class_pred_local[:,1])
    roc_auc = metrics.roc_auc_score(labels_local, class_pred_local[:,1])
    labels = np.where(class_pred_local[:,1]>=0.5, 1, 0)
    print(local_pato[i])
    print('--ROC SCORE %.4f'%(roc_auc))
    print('--AVG score %.4f'%(average_precision))
    TP, FP, TN, FN = perf_measure(labels_local, labels)
    print('TP, FP, TN, FN', TP, FP, TN, FN)
    se = TP/(TP+FN)
    sp = TN/ (TN+FP)
    print('--Sensitivity: %.4f'%(se))
    print('--Specificity: %.4f'%(sp))
    fpr, tpr, _ = roc_curve(labels_local, class_pred_local[:,1])
    se95, idx = find_nearest(tpr, 0.95)
    sp95 = 1 - fpr[idx]
    print('--Specificity 95: %.4f'%(sp95))

print('GLOBAL PATHOLOGIES')
global_pato=['Mala_diferenciacion', 'Hipercogenica']
#
for i in range(2):
    
    labels_global = class_true_global[:,i]
    #get the indexes with the actual pathology 
    indexes = indexes = np.where((labels_global==1) | (class_true==0))
    
    labels_global = labels_global[indexes]
    class_pred_global = class_pred [indexes]
    average_precision = metrics.average_precision_score(labels_global, class_pred_global[:,1])
    roc_auc = metrics.roc_auc_score(labels_global, class_pred_global[:,1])
    labels = np.where(class_pred_global[:,1]>=0.5, 1, 0)
    print(global_pato[i])
    print('--ROC SCORE %.4f'%(roc_auc))
    print('--AGV score %.4f'%(average_precision))
    TP, FP, TN, FN = perf_measure(labels_global, labels)
    se = TP/(TP+FN)
    sp = TN/ (TN+FP)
    print('--Sensitivity: %.4f'%(se))
    print('--Specificity: %.4f'%(sp))
    fpr, tpr, _ = roc_curve(labels_global, class_pred_global[:,1])
    se95, idx = find_nearest(tpr, 0.95)
    sp95 = 1 - fpr[idx]
    print('--Specificity 95: %.4f'%(sp95))
