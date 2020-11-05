#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:55:50 2020

@author: mpostigo
"""

import os
import torch
import random
import torchvision.transforms.functional as FT
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

def load_mean_std (data_dir, fold):
    """
    Load the mean and std from splits files which contains that information
    """
    mean=sio.loadmat(os.path.join(data_dir,'splits','rgbM'+str(fold)+'.mat'))['rgb_mean']
    std=sio.loadmat(os.path.join(data_dir,'splits','stdM'+str(fold)+'.mat'))['rgb_cov']
    std=np.sqrt(np.diag(std))
    print("Mean", mean)
    print("std", std)
    
    return mean, std
    
def collate_fn(batch):
    
    return tuple(zip(*batch))

def showBB (image, target, title, only_local = True):
    """
    Plot an image with the corresponding bounding boxes
    Args:
        image: the image which is plotted (ndarray)
        target: the information of the image (contains mask, labels and bounding boxes)(dict)
        title: the title that is given to the plot (string)
        only_local: boolean, defines if we are training with local or local+global pathologies
    """
    boxes = target['boxes']
    labels = target ['labels']
    
    #select the color map according to local or local+global pathologies
    if (only_local):
        label_map = {"Background":0, "Kidney": 1, "Cyst":2, "Pyramid":3, "Hydronephrosis":4, "Others":5}
        distinct_colors = ['#000080','#FFFFFF','#e6194b', '#3cb44b', '#ffe119', '#0082c8']
        label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
    else:
        label_map = {"Background":0, "Healthy": 1, "Cyst":2, "Pyramid":3, "Hydronephrosis":4, "Others":5, 
                     "PCD":6, "HC":7}
        distinct_colors = ['#000080','#FFFFFF','#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4']
        label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
        
    # Display the image
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.imshow(image)
    #uncomment tp show the mask too
    # plt.imshow(target['mask'], alpha = 0.3, cmap='gray')    
    
    #Draw the bounding boxes
    for box, label in zip(boxes, labels):
        
        if(type(box).__module__ == np.__name__):
          np_box = box
        else:
          np_box = box.numpy()
          
        #write the bounding box coordinates properly to be plotted  
        xmin = np_box[0]
        ymin = np_box[1]
        xmax = np_box[2]
        ymax = np_box[3]
        w = xmax-xmin
        h= ymax-ymin

        if(type(label).__module__ == np.__name__):
          np_label = label
        else:
          np_label = label.numpy()
          
        print('LABEL:', np_label)
        key = [key  for (key, value) in label_map.items() if value == np_label]
        print('KEY:', key)
        # Create a Rectangle patch
        color = label_color_map[key[0]]
        rect = patches.Rectangle((xmin, ymin),w ,h ,linewidth=3,edgecolor= color,facecolor='none', label='Label')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(xmin, ymin, key[0], fontsize=14, bbox=dict(facecolor=color, alpha=0.8))
        
    plt.title(title)
    plt.show()
    
def showBB_1label (image, target, title, only_local, gtlabel):
    """
    Args:
        image: the image which is plotted (ndarray)
        target: the information of the image (contains mask, labels and bounding boxes)(dict)
        title: the title that is given to the plot (string)
        only_local: boolean, defines if we are training with local or local+global pathologies
    """
    boxes = target['boxes']
    labels = target ['labels']
    indexesOneClass = np.where(labels==gtlabel)
    labels = labels[indexesOneClass]
    boxes = boxes[indexesOneClass]
    
    #select the color map according to local or local+global pathologies
    if (only_local):
        label_map = {"Background":0, "Kidney": 1, "Cyst":2, "Pyramid":3, "Hydronephrosis":4, "Others":5}
        distinct_colors = ['#000080','#FFFFFF','#e6194b', '#3cb44b', '#ffe119', '#0082c8']
        label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
    else:
        label_map = {"Background":0, "Healthy": 1, "Cyst":2, "Pyramid":3, "Hydronephrosis":4, "Others":5, 
                     "Bad_corticomedullary_differentiation":6, "Hyperechogenic_renal-cortex":7}
        distinct_colors = ['#000080','#FFFFFF','#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4']
        label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
        
    # Display the image
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.imshow(image)
    # plt.imshow(target['mask'], alpha = 0.3, cmap='gray')    
    
    #Draw the bounding boxes
    for box, label in zip(boxes, labels):
        
        if(type(box).__module__ == np.__name__):
          np_box = box
        else:
          np_box = box.numpy()
          
        #write the bounding box coordinates properly to be plotted  
        xmin = np_box[0]
        ymin = np_box[1]
        xmax = np_box[2]
        ymax = np_box[3]
        w = xmax-xmin
        h= ymax-ymin

        np_label = label.numpy()
        print('LABEL:', np_label)
        key = [key  for (key, value) in label_map.items() if value == np_label]
        print('KEY:', key)
        # Create a Rectangle patch
        color = label_color_map[key[0]]
        rect = patches.Rectangle((xmin, ymin),w ,h ,linewidth=1,edgecolor= color,facecolor='none', label='Label')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(xmin, ymin, key[0], bbox=dict(facecolor=color, alpha=0.2))
        
    plt.title(title)
    plt.show()

"""
CODE FROM: 
    https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=UYDb7PBw55b-
with some modifications for the custom dataset
"""
import math
from collections import defaultdict, deque
import time
import torch.distributed as dist
import datetime

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
        
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)   
        
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    
    epoch_dict = dict()
    loss_classifier = []
    loss_box_reg = []
    loss_objectness = []
    loss_rpn_box_reg = []
    
    model.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        

        loss_value = losses_reduced.item()
        
        loss_classifier.append(loss_dict_reduced.get('loss_classifier'))
        loss_box_reg.append(loss_dict_reduced.get('loss_box_reg'))
        loss_objectness.append(loss_dict_reduced.get('loss_objectness'))
        loss_rpn_box_reg.append(loss_dict_reduced.get('loss_rpn_box_reg'))
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    lc = np.mean([loss.cpu().detach().numpy() for loss in loss_classifier])
    lbr = np.mean([loss.cpu().detach().numpy() for loss in loss_box_reg])
    lo = np.mean([loss.cpu().detach().numpy() for loss in loss_objectness])
    lrbr = np.mean([loss.cpu().detach().numpy() for loss in loss_rpn_box_reg])

    return lc, lbr, lo, lrbr
  
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
