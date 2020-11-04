#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:53:48 2020

@author: mpostigo
"""

from sklearn.metrics import precision_score,f1_score
import sklearn.metrics as metrics
from torchvision import transforms, utils
import torch
import matplotlib.pyplot as plt
import time
import copy
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from DataLoadFunctions import *
from CustomTransforms import *
from CustomDataset import *
from tqdm import trange
from EarlyStopping import EarlyStopping
import torchio.transforms as tio



# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['label']
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    print(labels_batch)
    
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
#    inp = std.T[:, None] * inp + mean[:, None]
#    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def train_model(model , data_loader , criterion , 
                optimizer ,scheduler, device, dataset_sizes, n_classes, num_epochs=5):
    
  since = time.time()
  train_acc = []
  val_acc = []
  train_loss = []
  val_loss = []
  best_loss = np.inf
  best_model_wts = copy.deepcopy(model.state_dict())
  early_stopping = EarlyStopping(patience=7, verbose=True)

  for epoch in trange(num_epochs, desc="Epochs"):
    result = []
    for phase in ['Train', 'Val']:
      if phase=="Train":     # put the model in training mode
        model.train()
        
      else:     # put the model in validation mode
        model.eval()
      #Dataset size
      numSamples = dataset_sizes[phase]
      # Create variables to store outputs and labels
      outputs_m=np.zeros((numSamples, n_classes),dtype=np.float)
      labels_m=np.zeros((numSamples, n_classes),dtype=np.int)
      # keep track of training and validation loss
      contSamples=0
      running_loss = 0.0
      running_corrects = 0.0  
      
      for samples in data_loader[phase]:
        #load the data and target to respective device
        data, target = samples['image'].to(device)  , samples['label'].to(device)
        
        #Batch Size
        batchSize = target.shape[0]
                
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #forward
        with torch.set_grad_enabled(phase=="Train"):
          #feed the input
          output = model(data)
          # output = F.softmax(output)
          preds = torch.sigmoid(output).data > 0.5
          preds = preds.to(torch.float32)
          
          #calculate the loss
          loss = criterion(output, target)
          
          
          # backward + optimize only if in training phase
          if phase=="Train"  :
            # backward pass: compute gradient of the loss with respect to model parameters 
            loss.backward()
            # update the model parameters
            optimizer.step()

        # statistics
        running_loss += loss.item() * data.size(0)
        # Store outputs and labels 
        outputs_m [contSamples:contSamples+batchSize,...]=preds.cpu().numpy()
        labels_m [contSamples:contSamples+batchSize]=target.cpu().numpy()
        contSamples+=batchSize
        
      #Accumulated loss by epoch  
      epoch_loss = running_loss / len(data_loader[phase].dataset)  
      
      #At the end of an epoch, update the lr scheduler    
      if phase == 'Train':
        scheduler.step(epoch_loss)
        
      #Accumulated loss by epoch  
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      #Compute the AUCs at the end of the epoch
      aucs=computeAUCs(outputs_m,labels_m, n_classes)
      #And the Average AUC
      epoch_acc = aucs.mean()

      if (phase=="Val"):
        
        val_acc.append(epoch_acc)
        val_loss.append(epoch_loss)
        
        early_stopping(epoch_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            return model, train_acc, val_acc, train_loss, val_loss, best_model_wts
        
        scheduler.step(epoch_loss)        

        if (epoch_loss < best_loss):
          best_model_wts = copy.deepcopy(model.state_dict())
          print("model val_loss Improved from {:.8f} to {:.8f}".format(best_loss, epoch_loss))
          best_loss = epoch_loss
          
      if (phase=="Train"):
        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)


      result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
      print(result)
      
  # load best model weights
  model.load_state_dict(best_model_wts)    
  return model, train_acc, val_acc, train_loss, val_loss, best_model_wts


def visualize_model(model, device, dataloaders, num_images=6):
    """
    Function to visualize the different outputs of de model
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (samples) in enumerate(dataloaders['Val']):
            inputs = samples['image']
            labels = samples['label']
            inputs = inputs.to(device= device, dtype=torch.float)
            labels = labels.to(device= device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs).data > 0.5
            preds = preds.to(torch.float32)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                
                ax.set_title('P: %d ,%d, %d, %d ,%d, %d, %d || R: %d ,%d, %d, %d ,%d, %d %d'%(preds[j, 0],
                             preds[j, 1], preds[j, 2], preds[j, 3], preds[j, 4], preds[j, 5], preds[j, 6],
                             labels.cpu().data[j, 0], labels.cpu().data[j, 1],
                             labels.cpu().data[j, 2], labels.cpu().data[j, 3],
                             labels.cpu().data[j, 4], labels.cpu().data[j, 5],
                             labels.cpu().data[j, 6]))
                
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
       
def eval_performance (model, dataloaders, n_classes, device):
     """Evaluates the performance of the given model"""
     model.eval()
     total = 0

     y = np.zeros((0,n_classes),dtype=int)
     pred_y = np.zeros((0,n_classes),dtype=int)
     corrects = np.zeros(n_classes,)
    
     with torch.no_grad():
#        for i, (samples) in enumerate(dataloaders['Test']):
         # for i in range(4):
             for batch, (samples) in enumerate(dataloaders['Test']):
                inputs = samples['image']
                labels = samples['label']
                inputs = inputs.to(device= device, dtype=torch.float)
                labels = labels.to(device= device,dtype=torch.float)
        
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).data
                preds = torch.sigmoid(outputs).data > 0.5
                preds = preds.to(torch.float32)
                total += labels.size(0)
                
                for i in range(n_classes):
                    corrects[i]+=(preds[:,i] == labels[:,i]).sum().item()
                
                y=np.concatenate((y,labels.cpu().numpy()))
                pred_y=np.concatenate((pred_y, probs.cpu().numpy()))
    
     return corrects, y, pred_y
 

def get_dataloaders (data_dir, fold, batch_size, res):
    
    mean, std = load_mean_std(data_dir, fold)
    
    data_transforms = {
        'Train': transforms.Compose([#MyRotationTransform(),
                                                   MyHorizontalFlipTransform(),
                                                   applyMaskTransform(60,res),
                                                   ToTensor(), 
                                                   Normalize(mean, std)]),
    
        'Val': transforms.Compose([applyMaskTransform(60,res),
                                                   ToTensor(), 
                                                   Normalize(mean, std)]),
    
        'Test': transforms.Compose([applyMaskTransform(60,res),
                                                   ToTensor(), 
                                                   Normalize(mean, std)])  
    }
    
    
    image_datasets = {x: KidneyMaskDataset(data_dir = data_dir, partition = x, 
                                                          fold = fold, annotations=True, transform = data_transforms[x])
                      for x in ['Train', 'Val', 'Test']}
    
    labels = np.array(image_datasets['Train'].labels, dtype=np.int64)
    weigths = torch.tensor([len(labels)/np.sum(labels[:,0]==1), len(labels)/np.sum(labels[:,1]==1), len(labels)/np.sum(labels[:,2]==1), 
                            len(labels)/np.sum(labels[:,3]==1), len(labels)/np.sum(labels[:,4]==1), len(labels)/np.sum(labels[:,5]==1),])
    


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=False)
                  for x in ['Val','Test']}
    
    dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=batch_size,
                                                    drop_last=True)
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val','Test']}
    
    return image_datasets, dataloaders, dataset_sizes


       
def evaluate_model (image_datasets, dataloaders, dataset_sizes, n_classes, device, model_ft, criterion, optimizer, scheduler, num_epochs=25):


    """
    2. VISUALIZING DATA (ITERATING THROUGH THE DATASET)
    """
    dataset_sizes = {'Train': len(image_datasets['Train']), 'Val': len(image_datasets['Val'])}
    
    for i in range(len(image_datasets['Train'])):
        sample = image_datasets['Train'][i]
    
        print(i, sample['image'].size(), sample['mask'].size(), sample['label'])
    
        if i == 3:
            break
    
    for i_batch, sample_batched in enumerate(dataloaders['Val']):
        print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size(),
              sample_batched['label'].size())
        
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
    
    """
    3. TRAINING THE MODEL 
    """
    """--> Finetuning"""
      
    model_ft = model_ft.to(device)

    print('fine_tuning_trained_model')
    model_ft, train_acc, val_acc, train_loss, val_loss, best_model_wts = train_model(model_ft, dataloaders, criterion, optimizer, scheduler, 
                           device, dataset_sizes, n_classes, num_epochs=num_epochs)

    model_ft.load_state_dict(best_model_wts)
    visualize_model(model_ft, device, dataloaders)

    return model_ft, train_acc, val_acc, train_loss, val_loss, best_model_wts


                
def create_head(num_features , number_classes ,dropout_prob=0.5 ,activation_func =nn.ReLU):
  features_lst = [num_features , num_features//2 , num_features//4]
  layers = []
  for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
    layers.append(nn.Linear(in_f , out_f))
    layers.append(activation_func())
    layers.append(nn.BatchNorm1d(out_f))
    if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
  layers.append(nn.Linear(features_lst[-1] , number_classes))
  return nn.Sequential(*layers)

def computeAUCs(scores, labels, n_classes):
    #0: benign nevus, 1: malignant melanoma, 2: seborrheic keratosis.            
    aucs = np.zeros((n_classes,))
    for i in range(n_classes):
        aucs[i] = metrics.roc_auc_score(labels[:,i], scores[:,i])
    
    return aucs