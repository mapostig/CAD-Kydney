#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:25:27 2020

@author: mpostigo
Once all the results are saved: 
    Local object detection predictions
    Multi-label global image classification
    Multi-label local image classification

These results can be evaluated with the following code (another main)
"""

import torch
import os
from utils import *
from CustomDataset import KidneyDataset
from CustomTransforms import *
from Evaluation import *
import matplotlib.pyplot as plt
from bbox_utils import mask_multilabel_classification
from evaluate_results import create_detection_results_files, write_results_multiclass, save_outputs, read_output_files
from scipy import ndimage
from shutil import copyfile
from sklearn import metrics
from scipy import interp
from itertools import cycle
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import seaborn as sns
from scipy.special import softmax, expit, logit
from sklearn.metrics import roc_curve


"""
Main
"""
#1. Generate results from the trained model
generate_results ()

#2. From these detection results obtain the multi-label local classification
data_dir = '/home/mpostigo/Documentos/bbdd/output_results/'   
aggregations = ['mean', 'max', 'sum', 'log', 'area']    
for aggregation in ["max", "area", "sum", "log", "mean"]
    test_detection_multilabel(data_dir, approach=aggregation)
    
data_dir = "/home/mpostigo/Documentos/bbdd/resutls_FasterRCNN_multiclass/"
multi_dir ='/home/mpostigo/Documentos/bbdd/results/'

#To tets the best aggregation and approach
for aggregation in ["max", "area", "sum", "log", "mean"]:
    approach = "hybrid" #approach can be multi, detection and hybrid
    print(aggregation)
    compute_metrics_detection_multilabel (data_dir, aggregation, multi_dir, approach, 3/1.9)

aggregation = "max"
# for approach in ["detection", "multi", "hybrid"]:
#     print(approach)
    # for combinePato in ["sum", "max"]:
        # print(combinePato)
# for alpha in best_alphas:
compute_binary_SP95 (data_dir, aggregation, multi_dir, approach="hybrid", combinePato="max", alpha= 3/1.9)



"""
TO FIND THE BEST PARAMETERS, BEST ALPHA
detection_dir = "/home/mpostigo/Documentos/bbdd/resutls_FasterRCNN_multiclass/"
multi_dir ='/home/mpostigo/Documentos/bbdd/results/'

alphas = np.arange(0, 2, 0.001)
aggregations = ["max"]
# aggregations = ["max", "area"]
best_alphas = []
for fold in [1, 2, 3, 4, 5]:
    print("FOLD",fold)
    max_AUC, best_alpha, best_aggregation = findBestParamsHybrid (detection_dir, multi_dir, alphas, aggregations, fold)
    best_alphas.append(best_alpha)
    print("max_AUC, best_alpha, best_aggregation", max_AUC, best_alpha, best_aggregation)
"""
#to make scores comparable
def score_scale_fun(X, scores_old_min, scores_old_max, scores_new_min=0, scores_new_max=1):
    X = scores_new_max - ((scores_new_max - scores_new_min) * (scores_old_max - X) / (scores_old_max - scores_old_min))
    return X


def find_nearest(array, value):
    """
    To calculate the SP-95
    Parameters
    ----------
    array : ROC curve
    value : 0.95

    Returns
    -------
    The SE-95 
    idx : position
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def perf_measure(y_actual, y_hat):
    """To obatin True positives, false positives, true negatives, false negatives"""
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

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )
    
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_xlabel('True label')
        axes.set_ylabel('Predicted label')
        axes.set_title(class_label)

def generate_results (): 
    """
    Parameters
    ----------
    PATH : string with the path containing the trained model
    PATH = '/home/mpostigo/Documentos/bbdd/models/FasterRCNN_fold'+str(fold)+'batchSize'+str(batch_size)+'numEpochs'+str(num_epochs)+'HF_LR'+str(lr)+'.pt'

    Returns
    -------
    None.
    """
    folds = [1,2,3,4,5]
    
    start_id = 0
    accs = []
    batch_size= 1
    num_epochs = 30
    lr = 0.005
    
    for fold in folds: 
        
        print("Generating results for fold %d"%(fold))
        root= os.curdir
        PATH = '/home/mpostigo/Documentos/bbdd/models/FasterRCNN_fold'+str(fold)+'batchSize'+str(batch_size)+'numEpochs'+str(num_epochs)+'HF_LR'+str(lr)+'.pt'
        """
        1. Load the model
        """
        model = torch.load(PATH)
        
        """
        2.Create the test_loader
        """
        workers = 0
        mean, std = load_mean_std (root, fold)
        test_transforms = Compose([ToTensor(), Normalize(mean,std)])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        only_local = False
        test_dataset = KidneyDataset(root, 'Test', fold, transforms = test_transforms, only_local=only_local)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                                  collate_fn=collate_fn)
        idx = 0
        dst = '/home/mpostigo/Documentos/bbdd/output_results/images/'
        for file in test_dataset.imgs_files:
            copyfile(file, dst+'image_'+str(idx+start_id)+'.jpg')
            idx+=1
        """
        3. Generate the results
        """
        acc, test_images, test_targets, test_outputs = evaluate(model, 
                                            test_loader, device=device, th=0.5, phase='test')
        accs.append(acc)    
        """
        4. Visualize some results
        """
        print('Accuracy over the test set', acc)
        #
    
        save_outputs (test_targets, test_outputs, start_id, only_local = only_local)
        
        """SAVE THE BOUNDING BOX DETECTIONS TO DO THE MAP FROM  https://github.com/Cartucho/mAP"""
        create_detection_results_files (test_images, test_targets, 
                                    test_outputs, start_id, only_local = only_local)
        start_id+=len(test_dataset.imgs_files)

def test_detection_multilabel(data_dir, approach = 'max', th = 0.5):
    """
    Parameters
    ----------
    data_dir = '/home/mpostigo/Documentos/bbdd/output_results/'
    approach : mean, max, area, LSE...

    Returns
    -------
    None.
    """
    num_classes=7
    class_names = ("Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Bad_corticomedullary_differentiation", "Hyperechogenic_renal_cortex")
    
    #Classification auc over the classes:
    GT = []
    Pred = []
    test_images, test_targets, test_outputs = read_output_files (data_dir)
    
    counter = 15
    for j, (image, target, output) in enumerate(zip(test_images, test_targets, test_outputs)):

            """
            APPLY AGGREGATION ON THE DETECTIONS: SEE 3.2.3.2 IN TFM_MariaPostigo.pdf
            """
            boxes_masks_per_label, gt_multiclass, pred_multiclass, unique_pred_labels = mask_multilabel_classification (image, target['boxes'],
                                            target['labels'], output['boxes'], output['labels'], 
                                            output['scores'], num_classes, approach, th)
            
            GT.append(gt_multiclass)
            Pred.append(pred_multiclass)
            n_labels = len(unique_pred_labels)
            
            while (counter>0 and n_labels>0):
                
                plt.imshow(image)
                
                showBB(image, target, ('GT Boxes %d')%(j), only_local=False)
                
                for i, label in enumerate(unique_pred_labels):
                    n_labels-=1
                    
                    plt.figure()
                    plt.title((class_names[label-1]+' %.4f id %d')%(np.amax(boxes_masks_per_label[i]), i))

                    plt.imshow(image)
                    blurred_mask = ndimage.gaussian_filter(boxes_masks_per_label[i], sigma=3)
                    plt.imshow(blurred_mask, alpha = 0.2, cmap='jet')
                    plt.clim(0, 1)
                    plt.colorbar()
                    plt.show()
                    
                plt.imshow(image)    
                blurred_mask = ndimage.gaussian_filter(np.sum(boxes_masks_per_label, axis=0), sigma=3)
                plt.imshow(blurred_mask, alpha = 0.2, cmap='jet')
                plt.show()
                counter-=1

          
    results_dir = "/home/mpostigo/Documentos/bbdd/resutls_FasterRCNN_multiclass"
    
    """SAVE THE MULTI-LABEL CLASSIFICATION FROM OBJECT DETECTION"""
    write_results_multiclass (GT, Pred, results_dir, approach)    

    # for element in dir():
    
    #     if (element[0:2] != "__") and ('start_id' not in element) and ('fold' not in element) and ('acc' not in element):
    #         del globals()[element]
    
    # import torch
    
    # torch.cuda.empty_cache()

"""
COMPUTE ROC CURVES
code from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
 """
def compute_metrics_detection_multilabel (data_dir, aggregation, multi_dir, approach, alpha, showPlots = False):
    """
    Parameters
    ----------
    data_dir : directory with the detection results
    aggregation : string in ["max", "area", "sum", "log", "mean"] according to the 
                aggregation method used for the probabilities from detection
    multi_dir : directory with the multilabel classification results
    approach : string in "detection", "multi", "hybrid"
    showPlots : boolean to show the plots

    Returns
    -------
    None.
    """
    #1 Load the data    
    if approach == "detection":
        y_test = np.loadtxt(data_dir+"/GT_"+aggregation+".txt")
        y_score = np.loadtxt(data_dir+"/pred_"+aggregation+".txt")
        #we use the scores (probabilities) as they come from the aggregation method. No sigmoid.
    elif approach == "multi":
        y_test = np.loadtxt(multi_dir+"/GT.txt")
        y_score = np.loadtxt(multi_dir+"/pred.txt")
        #we use the scores (probabilities) as they come from the multilabel classification. Already sigmoids.
    else: #hybrid case
        y_test = np.loadtxt(data_dir+"/GT_"+aggregation+".txt")
        y_score_det = np.loadtxt(data_dir+"/pred_"+aggregation+".txt")
        y_score_multi = np.loadtxt(multi_dir+"/pred.txt")
        
        if aggregation in ["area", "sum", "log"]: #in this cases the scores arent probabilities, so we normalize them by fold 1 (397 elements)
            max_scores = np.amax(y_score_det[:397,:], axis=0) #max value per colum
            y_score_det = y_score_det/max_scores
            
        y_score = np.add(y_score_multi*1.9, y_score_det*3)

    AUCS = []
    n_classes = 7
    class_names = ["Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Poor Corticomedular Differenciation", "Hyperechogenic cortex"]
    
    labels = ["HE", "C", "P", "HY", "O", "pcd", "hc"]
    
    if showPlots:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            
            precision[i], recall[i], _ = metrics.precision_recall_curve(y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = metrics.average_precision_score(y_test[:, i], y_score[:, i])
    
        # Plot all ROC curves
        plt.figure(figsize=(9, 8))
        ax = plt.subplot(111)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy', 'green', 'black'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(class_names[i], roc_auc[i]))
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        ax.legend(loc=(-0.1, -.50) , bbox_to_anchor=(-0.05, -0.45), prop=dict(size=14))
        plt.show()
        
        # Plot all PRECISION RECALL curves
        plt.figure(figsize=(9, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        
        lines.append(l)
        labels.append('iso-f1 curves')
        
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy', 'green', 'black'])
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(class_names[i], average_precision[i]))
        
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(-0.1, -.50), bbox_to_anchor=(-0.05, -0.6), prop=dict(size=14))
        
        plt.show()
        # y_score_labels =  np.where(y_score>0.5, 1, 0)
    
        fig, ax = plt.subplots(2, 4, figsize=(15, 7))
        
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
            print_confusion_matrix(cfs_matrix, axes, label, ['N', 'Y'])
        
        # fig.tight_layout()
        plt.show()  

    for i, label in enumerate(labels):
        # print(label)
        average_precision = metrics.average_precision_score(y_test[:,i], y_score[:,i])
        roc_auc = metrics.roc_auc_score(y_test[:,i], y_score[:,i])
        print('--ROC SCORE %.4f'%(roc_auc))
        print(roc_auc)
        AUCS.append(roc_auc)
        print('--AVG score %.4f'%(average_precision))
        print(average_precision)
        TP, FP, TN, FN = perf_measure(y_test[:,i], y_score_labels[:,i])
        print('TP, FP, TN, FN', TP, FP, TN, FN)
        se = TP/(TP+FN)
        sp = TN/ (TN+FP)
        print('--Sensitivity: %.4f'%(se))
        print('--Specificity: %.4f'%(sp))
        
        fpr, tpr, _ = roc_curve(y_test[:,i], y_score[:,i])
        se95, idx = find_nearest(tpr, 0.95)
        sp95 = 1 - fpr[idx]
        print('--Specificity 95: %.4f'%(sp95))
        print(se)
        print(sp95)
    # print("alpha", alpha)    
    print("Mean AUC: ", np.mean(AUCS))

def compute_binary_SP95 (data_dir, aggregation, multi_dir, approach, combinePato, alpha):
    """
    Parameters
    ----------
    data_dir : directory with the detection results
    aggregation : string in ["max", "area", "sum", "log", "mean"] according to the 
                aggregation method used for the probabilities from detection
    multi_dir : directory with the multilabel classification results
    approach : string in "detection", "multi", "hybrid"
    combinePato : How to combine the pathologies column in a single one (by max or by sum)

    Returns
    -------
    None.
    """
    #1 Load the data    
    
    if approach == "detection":
        y_test = np.loadtxt(data_dir+"/GT_"+aggregation+".txt")
        y_score = np.loadtxt(data_dir+"/pred_"+aggregation+".txt")
        #we use the scores (probabilities) as they come from the aggregation method. No sigmoid.
    elif approach == "multi":
        y_test = np.loadtxt(multi_dir+"/GT.txt")
        y_score = np.loadtxt(multi_dir+"/pred.txt")
        #we use the scores (probabilities) as they come from the multilabel classification. Already sigmoids.
    else: #hybrid case
        y_test = np.loadtxt(data_dir+"/GT_"+aggregation+".txt")
        y_score_det = np.loadtxt(data_dir+"/pred_"+aggregation+".txt")
        y_score_multi = np.loadtxt(multi_dir+"/pred.txt")
        
        if aggregation in ["area", "sum", "log"]: #in this cases the scores arent probabilities, so we normalize them by fold 1 (397 elements)
            max_scores = np.amax(y_score_det[:397,:], axis=0) #max value per colum
            y_score_det = y_score_det/max_scores
            
        y_score = np.add(y_score_multi, y_score_det*alpha)
    
    y_score_healthy = y_score[:,0]
    if combinePato == "sum":
        y_score_sick =  np.sum(y_score[:,1:], axis=1)
    else:
        y_score_sick =  np.amax(y_score[:,1:], axis=1)
        
    #create a binary label 0 healthy 1 sick
    labels = 1 - y_test[:,0]
    y_score_binary = np.zeros((len(y_score), 2))
    
    y_score_binary[:,0] = y_score_healthy
    y_score_binary[:,1] = y_score_sick
    
    # max_binary = np.amax(y_score_binary)
                         
    y_score_binary = softmax(y_score_binary, axis=1)
    print(alpha)
    fpr, tpr, _ = metrics.roc_curve(labels, y_score_binary[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)
    se95, idx = find_nearest(tpr, 0.95)
    sp95 = 1 - fpr[idx]
    print("SP95", sp95)
    


def findBestParamsHybrid (detection_dir, multi_dir, alphas, aggregations, fold = 1):
    
    fold_size = 397
    end = fold*fold_size
    start = end-fold_size
    labels = ["HE", "C", "P", "HY", "O", "pcd", "hc"]
    max_AUC = 0
    best_alpha = 0
    best_beta = 0
    best_aggregation = ""
    
    for aggregation in aggregations:
        # print(aggregation)
        y_test = np.loadtxt(detection_dir+"/GT_"+aggregation+".txt")
        y_test = y_test[start:end, :]
        y_score_det = np.loadtxt(detection_dir+"/pred_"+aggregation+".txt")
        y_score_det = y_score_det[start:end, :]
        y_score_multi = np.loadtxt(multi_dir+"/pred.txt")
        y_score_multi = y_score_multi[start:end, :]
        
        # if aggregation in ["area", "sum", "log"]: #in this cases the scores arent probabilities, so we normalize them by fold 1 (397 elements)
        #     max_scores = np.amax(y_score_det, axis=0) #max value per colum
        #     y_score_det = y_score_det/max_scores
        
        for alpha in alphas:

            y_score = np.add(y_score_multi, y_score_det*alpha)
            AUCS = []
            for i, label in enumerate(labels):
                roc_auc = metrics.roc_auc_score(y_test[:,i], y_score[:,i])
                AUCS.append(roc_auc)
                
            # print(np.mean(AUCS))    
            if(np.mean(AUCS) > max_AUC):
                max_AUC = np.mean(AUCS)
                best_alpha = alpha
                best_aggregation = aggregation

                
    return max_AUC, best_alpha, best_aggregation



