#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:58:29 2020

@author: mpostigo

This is the main for custom object detection
"""

folds = [1, 2, 3, 4, 5]

for fold in folds:
    #import here since we free the memory in every fold
    import torch
    import os
    from utils import *
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    import matplotlib.pyplot as plt
    from CustomDataset import KidneyDataset
    from CustomTransforms import *
    import torchvision
    from Evaluation import *
    import scipy.io as sio
    import numpy as np
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from evaluate_results import *
    from EarlyStopping import EarlyStopping
    from bbox_utils import *
    
    root= os.curdir
    partition = 'Train'
    print('Creating dataset for fold ', fold)
    start_id=0
    only_local = False #We can do local pathologies detection od global and local detection
    
    dataset = KidneyDataset(root, partition, fold, transforms=None, only_local=only_local)
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                              SHOW IMAGES                                    %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    for i in range(100):
        image, target = dataset.__getitem__(i)
        print('index:',i)
    #   print('Boundin boxes: ', target['boxes'])
        plt.imshow(image)
        title = ('Image %d'%(i))
        showBB (image, target, title, only_local = only_local)

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                         COMPOSED TRANSFORMS                                 %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    flip = HorizontalFlipTransform()
    crop = MaskTransform(30)
    rot = RotationTransform ()
    brigth = RandomIncreaseBrightness ()
    noise = AddNoise()
    mean, std = load_mean_std (root, fold)
    norm = Compose([ToTensor(), Normalize(mean,std)])
    i=7
    

    # Apply each of the above transforms on sample.
    image, target = dataset.__getitem__(i)
    title='Original'
    showBB (image, target, title, only_local)
#    
#    for i in range(10):
#        image, target = dataset.__getitem__(7)
#        rot_image, rot_target = rot(image, target) 
#        title = 'Rotate' 
#        showBB (rot_image, rot_target, title, only_local)
    
    flip_image, flip_target = flip(image, target) 
    title = 'Horizontal Flip' 
    showBB (flip_image, flip_target, title, only_local)
    
    image, target = dataset.__getitem__(i)
    crop_image, crop_target = crop(image, target) 
    title = 'Crop By Mask' 
    showBB (crop_image, crop_target, title, only_local)
    
    image, target = dataset.__getitem__(i)
    rot_image, rot_target = rot(image, target) 
    title = 'Rotate' 
    showBB (rot_image, rot_target, title, only_local)
    
    image, target = dataset.__getitem__(i)
    brigth_image, brigth_target = brigth(image, target) 
    title = 'Brightness' 
    showBB (brigth_image, brigth_target, title, only_local)
    
    image, target = dataset.__getitem__(i)
    noise_image,noise_target = noise(image, target)
    title='Noise'
    showBB (noise_image, noise_target, title, only_local)
    
#    image, target = dataset.__getitem__(i)
#    norm_image, norm_target = norm(image, target)
#    title='Normalization'
#    norm_target['mask'] = norm_target['mask'].vflip(norm_target['mask'],axis=1).copy())
#    showBB (norm_image.numpy().transpose((1, 2, 0)), norm_target, title, only_local)#image[0].numpy().transpose((1, 2, 0)), target[0]
    
    image, target = dataset.__getitem__(i)
    compose = Compose([crop, flip, rot, brigth])
    t_image, t_target = compose(image, target) 
    title = 'Compose' 
    showBB (t_image, t_target, title, only_local)
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                             CREATE DATASETS                                 %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    #By different experiments we opted to use the HF as the unique data
    #augmentation method
    my_transforms = Compose([HorizontalFlipTransform(),
                                        ToTensor(),
                                        Normalize(mean,std)])
    
    test_transforms = Compose([ToTensor(), Normalize(mean,std)])
    
    train_dataset = KidneyDataset(root, 'Train', fold, transforms = my_transforms, only_local=only_local)
    test_dataset = KidneyDataset(root, 'Test', fold, transforms = test_transforms, only_local=only_local)
    val_dataset = KidneyDataset(root, 'Val', fold, transforms = test_transforms, only_local=only_local)
    
    
    print('Train dataset:', len(train_dataset))
    print('Val dataset:', len(val_dataset))
    print('Test dataset:', len(test_dataset))
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                          CREATE DATALOADERS                                 %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    batch_size = 1
    workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                                collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                                collate_fn=collate_fn)

        
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                        FINETUNING THE MODEL                                 %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    from torchvision.models.detection import FasterRCNN
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 
    num_classes = len(train_dataset.class_names)+1 #plus the background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    """
    The backbone can be modified, but resnet50 performance is good to solve
    the given problem
    """
#    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#    backbone.out_channels = 1280 
#    model = FasterRCNN(backbone,
#                   num_classes=num_classes)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    lr = 0.005
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                              TRAIN EPOCHS                                   %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    num_epochs = 25
    all_acc = []
    best_acc=0
    th = 0.5
    phase='val'

    #save fasterRCNN losses per epoch to see convergence
    loss_classifier = []
    loss_box_reg = []
    loss_objectness = []
    loss_rpn_box_reg = []

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        lc, lbr, lo, lrbr  = train_one_epoch(model, optimizer, train_loader, device, epoch, 10)
        
        loss_classifier.append(lc)
        loss_box_reg.append(lbr)
        loss_objectness.append(lo)
        loss_rpn_box_reg.append(lrbr)
        
        # update the learning rate
        lr_scheduler.step()
        #evaluate on the validation dataset
        acc, val_images, val_targets, val_outputs = evaluate(model, val_loader, device, th, phase)

        all_acc.append(acc) #TODO: use losses criterion
    
        print('Epoch accuracy: ', acc)
  
    plt.figure()
    plt.plot(range(num_epochs), loss_classifier, 'r', label = 'loss_classifier')
    plt.plot(range(num_epochs), loss_box_reg, 'b', label = 'loss_box_reg')
    plt.plot(range(num_epochs), loss_objectness, 'g', label = 'loss_objectness')
    plt.plot(range(num_epochs), loss_rpn_box_reg, 'orange', label = 'loss_rpn_box_reg')
    plt.legend()
    plt.ylabel('losses')
    plt.xlabel('epochs')
    plt.title('Train Loss')
    # plt.savefig('TrainlossesFold'+str(fold)+'num_epochs'+str(num_epochs)+'_T_HF_rot_batch_'+str(batch_size)+'lr_'+str(lr)+'_.png')

    plt.figure()
    plt.plot(range(num_epochs), all_acc, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Validation accuracy')
    plt.title('Accuracy evolution')
    plt.show()  
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                     MAP Mean Average Precision                              %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    # The MAP used is downloaded from another project:
    #git clone https://github.com/Cartucho/mAP
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                             % 
    %                       SAVE THE TRAINED MODEL                                %
    %                                                                             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    PATH = '/home/mpostigo/Documentos/bbdd/models/FasterRCNN_fold'+str(fold)+'batchSize'+str(batch_size)+'numEpochs'+str(num_epochs)+'HF_LR'+str(lr)+'.pt'
    # model = torch.load(PATH)
    """SAVE THE MODEL"""
    torch.save(model, PATH)
    
    





