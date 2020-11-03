#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:16:07 2020

@author: mpostigo

Here the training and evaluation methods

"""
import Custom_dataset
import Custom_transforms
from torchvision import transforms, utils
import torch
import matplotlib.pyplot as plt
import DataLoad_Functions
import time
import copy
import torch.nn as nn
from EarlyStopping import EarlyStopping
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

data_dir = '/home/mpostigo/Documentos/kidney/bbdd/'

# fold = 1 #1-5


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
    
    
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Function for training a given model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) #variable to save the best weigths for the model
    best_acc = 0.0
#    earlystop = EarlyStopping(patience = 10,verbose = True)
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training modeç

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for samples in dataloaders[phase]:
                inputs = samples['image']
                labels = samples['label']
                inputs = inputs.to(device= device, dtype=torch.float)
                labels = labels.to(device= device)
#                print('Labels in batch: %d health %d sick'%((labels==0).sum(), (labels==1).sum()))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                        
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} LR: {:.8f}'.format(
                phase, epoch_loss, epoch_acc, scheduler.get_last_lr()[0]))

            # deep copy the model
            if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'Val':
                val_acc.append(epoch_acc)
#                earlystop(epoch_acc,model)
            if phase == 'Train':
                train_acc.append(epoch_acc)
                
#        if(earlystop.early_stop):
#            print("Early stopping")
#            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc


def visualize_model(model, device, class_names, dataloaders,num_images=6):
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
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: %d, real %d'%(class_names[preds[j]], labels.cpu().data[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
       
def eval_performance (model, dataloaders, device):
     """Evaluates the performance of the given model"""
     model.eval()
     total = 0
     correct = 0

     y = np.zeros((0,),dtype=int)
     pred_y = np.zeros((0,2),dtype=int)
     local =  np.zeros((0,4),dtype=int)
     global_ =  np.zeros((0,2),dtype=int)
    
     with torch.no_grad():
#        for i, (samples) in enumerate(dataloaders['Test']):
         for i in range(4):
             for batch, samples in dataloaders['Test']:
                inputs = samples['image']
                labels = samples['label']
                global_a = samples['ga']
                local_a = samples['la']
                inputs = inputs.to(device= device, dtype=torch.float)
                labels = labels.to(device= device,dtype=torch.float)
                
                if batch == 3:
                    plt.figure()
                    show_landmarks_batch(sample_)
                    plt.axis('off')
                    plt.ioff()
                    plt.show()
                    
        
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
    #            print('%d out of %d at batch %d'%(correct,total,i))
                
                y=np.concatenate((y,labels.cpu().numpy()), axis=0)
                pred_y=np.concatenate((pred_y, F.softmax(outputs).cpu().numpy()),axis=0)
                
                local= np.concatenate((local, local_a.numpy()), axis=0)
                global_= np.concatenate((global_, global_a.numpy()), axis=0)
    
     return correct/total, y, pred_y, global_,local
 

def get_dataloaders (data_dir, fold, batch_size, res):
    
    mean, std = DataLoad_Functions.load_mean_std(data_dir, fold)
    
    data_transforms = {
        'Train': transforms.Compose([Custom_transforms.MyRotationTransform(),
                                                   Custom_transforms.MyHorizontalFlipTransform(),
                                                   Custom_transforms.applyMaskTransform(20,res),
#                                                   Custom_transforms.RandomIncreaseBrightness(),
                                                   Custom_transforms.ToTensor(), 
                                                   Custom_transforms.Normalize(mean, std)]),
    
        'Val': transforms.Compose([Custom_transforms.applyMaskTransform(20,res),
                                                   Custom_transforms.ToTensor(), 
                                                   Custom_transforms.Normalize(mean, std)]),
    
        'Test': transforms.Compose([Custom_transforms.MyRotationTransform(),
                                                   Custom_transforms.MyHorizontalFlipTransform(),
                                                   Custom_transforms.applyMaskTransform(20,res),
#                                                   Custom_transforms.RandomIncreaseBrightness(),
                                                   Custom_transforms.ToTensor(), 
                                                   Custom_transforms.Normalize(mean, std)])  
#        'Test': transforms.Compose([Custom_transforms.applyMaskTransform(20,res),
#                                                   Custom_transforms.ToTensor(), 
#                                                   Custom_transforms.Normalize(mean, std)])
    }
    
    
    image_datasets = {x: Custom_dataset.KidneyMaskDataset(data_dir = data_dir, partition = x, 
                                                          fold = fold, annotations=True, transform = data_transforms[x])
                      for x in ['Train', 'Val', 'Test']}
    
    target = np.array(image_datasets['Train'].labels, dtype=np.int64)
    cls_weights = torch.from_numpy(
        compute_class_weight('balanced', np.unique(target), target))
    weights = cls_weights[target]
    sampler = WeightedRandomSampler(weights, len(target), replacement=True)
    
             
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=False)
                  for x in ['Val','Test']}
    
    dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=batch_size,
                                                  sampler=sampler, drop_last=True)
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val','Test']}
    
    return image_datasets, dataloaders, dataset_sizes


       
def evaluate_model (image_datasets, dataloaders, dataset_sizes, device, model_ft, criterion, optimizer, scheduler, num_epochs=25):

    """
    1. LOAD THE DATA
    """
    #dataloaders passed as parameters
    class_names = [0, 1] #TODO: add a classes attribute at kidney dataset: self.classes = [healthy, sick]
    
    
    
    """
    2. VISUALIZING DATA (ITERATING THROUGH THE DATASET)
    """
    for i in range(len(image_datasets['Train'])):
        sample = image_datasets['Train'][i]
    
        print(i, sample['image'].size(), sample['mask'].size(), sample['label'])
    
        if i == 3:
            break
    """
        However, we are losing a lot of features by using a simple for loop to iterate over the data. 
        In particular, we are missing out on:
        
            Batching the data
            Shuffling the data
            Load the data in parallel using multiprocessing workers.
    """
    
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
    
    model_ft = model_ft.to(device)
    

    print('fine_tuning_trained_model')
    
    model_ft, train_acc, val_acc = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer, scheduler,device,
                           num_epochs=num_epochs)
    
    visualize_model(model_ft, device, class_names, dataloaders)
    """------------------------------------------------------------------------"""
    
#    for param in model_conv.parameters():
#        param.requires_grad = False
#    
#    # Parameters of newly constructed modules have requires_grad=True by default
#    num_ftrs = model_conv.fc.in_features
#    model_conv.fc = nn.Linear(num_ftrs, 2)
#    
#    model_conv = model_conv.to(device)
#    
#    print('feature_extractor_model')
#    model_conv = train_model(model_conv,dataloaders, dataset_sizes, criterion, optimizer,
#                             scheduler, device, num_epochs=num_epochs)
    """
    As PyTorch's documentation on transfer learning explains, there are two major ways that transfer learning is used: 
        -Fine-tuning a CNN or by using the CNN as a fixed feature extractor. 
        When fine-tuning a CNN, you use the weights the pretrained network has instead of randomly initializing them, 
        and then you train like normal. 
        -In contrast, a feature extractor approach means that you'll maintain all the weights of the CNN except for those
        in the final few layers, which will be initialized randomly and trained as normal.
        Fine-tuning a model is important because although the model has been pretrained, it has been trained on a different 
        (though hopefully similar) task. The densely connected weights that the pretrained model comes with will probably 
        be somewhat insufficient for your needs, so you will likely want to retrain the final few layers of the network
    """

    return model_ft, train_acc, val_acc


                

