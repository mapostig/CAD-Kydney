#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:25:14 2020

@author: mpostigo

Script to try different models over different folds

"""
import Custom_models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch
import matplotlib.pyplot as plt
import numpy as np 
from efficientnet_pytorch import EfficientNet
import scipy.io as sio



data_dir = '/home/mpostigo/Documentos/kidney/bbdd/'
folds = [1,2,3,4,5] #1-5
path_to_save_model = '/home/mpostigo/Documentos/Models/resnet50/'

for fold in folds:
    
    print('Executing fold',fold)
    #tf_efficientnet_b7: RandAugment trained model with B7 backbone (res: 600)
    #resnet res=224
    res = 224
    #Get the dataloaders
    batch_size = 32
    image_datasets, dataloaders, dataset_sizes = Custom_models.get_dataloaders (data_dir, fold, batch_size, res)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Train models
  
    model = models.resnet50(pretrained=True)
##    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
##    model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_l2_ns_475', pretrained=True, num_classes=2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
#    model= models.alexnet(pretrained=True)
##    set_parameter_requires_grad(model_ft, feature_extract)
    num_classes= 2
#    num_ftrs = model.classifier[6].in_features
#    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    model= model.to(device)
    # weights = [1-image_datasets['Train'].labels.count(0)/dataset_sizes['Train'], 
    #           1-image_datasets['Train'].labels.count(1)/dataset_sizes['Train']]
    # class_weights = torch.FloatTensor(weights).cuda()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    criterion = nn.CrossEntropyLoss()
    lr =  0.01
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
    n_epochs = 100
    """
    TRAIN THE MODEL
    """
    finetuned_model, train_acc, val_acc = Custom_models.evaluate_model (image_datasets, dataloaders, dataset_sizes, device, model,
                                        criterion, optimizer, scheduler, num_epochs=n_epochs)
   
    plt.figure()
    plt.plot(train_acc, label = 'Train acc')
    plt.plot(val_acc, label  = 'Val acc' )
    plt.legend()
    plt.ylabel('Accuracies')
    plt.xlabel('Epocs')
    plt.grid()
    

    torch.save(finetuned_model.state_dict(), 
    path_to_save_model+'batch_'+str(batch_size)+'lr_'+str(lr)+'_resnet_foldM'+str(fold)+'.pth')
    print('saved '+path_to_save_model+'batch_'+str(batch_size)+'lr_'+str(lr)+'_resnet_foldM'+str(fold)+'.pth')

    """
    TEST THE MODEL
    """
    device = torch.device("cuda")

    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
#    
#    model= models.alexnet(pretrained=True)
#    num_classes= 2
#    num_ftrs = model.classifier[6].in_features
#    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    
    
    model.load_state_dict(torch.load(path_to_save_model+'batch_'+str(batch_size)+'lr_'+str(lr)+'_resnet50_foldM'+str(fold)+'.pth'))
#    model.load_state_dict(torch.load(path_to_save_model+'batch_32lr_0.1_tf_efficientnet_b1_foldM'+str(fold)+'_weighted.pth'))
    model.to(device)
#    
    acc, labels, pred_labels, global_a, local_a= Custom_models.eval_performance (model, dataloaders, device)
    
