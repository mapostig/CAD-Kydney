#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:35:18 2020

@author: mpostigo

Main workflow for multi-label global image classification
"""

start_id = 0
folds = [1,2,3,4,5] #1-5
#Execute the multi-label image classification for the 5 folds

for fold in folds:
    #Some libraries are imported here since we free the memory after training each fold
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from CustomDataset import *
    from CustomTransforms import *
    from Utils import *
    from DataLoadFunctions import *
    
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models
    import torch
    from evaluate_results import *
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx
    
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
    
    data_dir ='/home/mpostigo/Documentos/bbdd'
    
    """
    Multi-label image classification
    IN THIS CLASSIFICATION EACH IMAGE CAN BELONG TO 7 DIFFERENT CLASSES 
    THAT ARE NOR MUTUALLY EXCLUSIVE, EXCEPT THE HEALTHY LABEL
    """
    
    class_names = ["Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Poor Corticomedular Differenciation", "Hyperechogenic cortex"]
    
    print('Executing fold',fold)
    res = 224
    #Get the dataloaders
    batch_size = 32
    image_datasets, dataloaders, dataset_sizes = get_dataloaders (data_dir, fold, batch_size, res)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Train model

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features    
    num_classes= 7 #1 healthy, 2 global pathologies and 4 local pathologies
    top_head = create_head(num_ftrs , num_classes) 
    model.fc = top_head # replace the fully connected layer

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.09)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 2)
    n_epochs = 50
    
    finetuned_model, train_acc, val_acc, train_loss, val_loss, best_model_wts = evaluate_model (image_datasets, 
                                        dataloaders, dataset_sizes, num_classes, device, model,
                                       criterion, optimizer, scheduler, num_epochs=n_epochs)

    plt.figure()
    print(train_acc)
    print(val_acc)
    plt.plot(train_acc, label = 'Train auc')
    plt.plot(val_acc, label  = 'Val auc' )
    plt.legend()
    plt.ylabel('AUCS')
    plt.xlabel('Epocs')
    plt.title('Fold %d'%(fold))
    plt.grid()
    
    plt.figure()
    print(train_loss)
    print(val_loss)
    plt.plot(train_loss, label = 'Train loss')
    plt.plot(val_loss, label  = 'Val loss' )
    plt.legend()
    plt.ylabel('Losses')
    plt.xlabel('Epocs')
    plt.title('Fold %d'%(fold))
    plt.grid()

    corrects, y_test, y_score= eval_performance (finetuned_model, dataloaders, num_classes, device)

    for i, name in enumerate(class_names):
        print("Test auc %s: %.4f"%(name, corrects[i]))

    write_results (y_test, y_score, data_dir+"/results", start_id)
    start_id+=1
    
    #Clean the memory for every fold
    for element in dir():

        if (element[0:2] != "__") and ('start_id' not in element) and ('fold' not in element):
            del globals()[element]
    
    import torch
    
    torch.cuda.empty_cache()
    

"""
COMPUTE ROC CURVES
code from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
 """
#1 Load the results and the GT previously saved
data_dir ='/home/mpostigo/Documentos/bbdd'
results_dir = data_dir+'/results'

gt_file = results_dir+"/GT.txt"
pred_file_multi = results_dir+"/pred.txt"

y_test = np.loadtxt(gt_file)
y_score = np.loadtxt(pred_file_multi)

#This is commented since is employed for the hybrid method
# results_dir2 = "/home/mpostigo/Documentos/bbdd/resutls_FasterRCNN_multiclass"
# y_score_det = np.loadtxt(results_dir2+"/pred_max.txt")
# #normalize
# # y_score_det_norm = y_score_det / y_score_det.max(axis=0)
# # row_sums = y_score_det.sum(axis=1)
# # y_score_det = y_score_det / row_sums[:, np.newaxis]
# # # y_score_det = y_score_det/np.amax(y_score) #addNorm

# # # # y_score_aux = np.multiply(y_score_multi, y_score_det)
# y_score = np.add(y_score_multi, y_score_det)
# # y_score =  y_score_multi

# y_score = np.where(y_score>1, 1, y_score) #Adding



"""
SEE STATISTICAL RESULTS
"""
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
from itertools import cycle

AUCS = []

# Compute ROC curve and ROC area for each class and precision recall
n_classes = 7
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

from sklearn.metrics import multilabel_confusion_matrix

y_score_labels =  np.where(y_score>0.5, 1, 0)

confusion_matrix = multilabel_confusion_matrix(y_test, y_score_labels)
print(confusion_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_score_labels))

fig, ax = plt.subplots(2, 4, figsize=(15, 7))
labels = ["HE", "C", "P", "HY", "O", "pcd", "hc"]
    
for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ['N', 'Y'])

# fig.tight_layout()
plt.show()  

def perf_measure(y_actual, y_hat):
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

for i, label in enumerate(labels):
    print(label)
    average_precision = metrics.average_precision_score(y_test[:,i], y_score[:,i])
    roc_auc = metrics.roc_auc_score(y_test[:,i], y_score[:,i])
    print('--ROC SCORE %.4f'%(roc_auc))
    AUCS.append(roc_auc)
    print('--AVG score %.4f'%(average_precision))
    TP, FP, TN, FN = perf_measure(y_test[:,i], y_score_labels[:,i])
    print('TP, FP, TN, FN', TP, FP, TN, FN)
    se = TP/(TP+FN)
    sp = TN/ (TN+FP)
    print('--Sensitivity: %.4f'%(se))
    print('--Specificity: %.4f'%(sp))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test[:,i], y_score[:,i])
    se95, idx = find_nearest(tpr, 0.95)
    sp95 = 1 - fpr[idx]
    print('--Specificity 95: %.4f'%(sp95))
    
    
print("Mean AUC: ", np.mean(AUCS))
