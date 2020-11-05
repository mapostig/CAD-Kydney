#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:03:59 2020

@author: mpostigo
"""
import numpy as np
import cv2
import math as m
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.draw import polygon
import torch
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def _get_masks_from_boxes(image, boxes):
    """
    Given a set of bounding boxes, their corresponding masks are created
    Args: 
        image: the image that contains the given boxes
        boxes: list of bounding boxes
    """
    boxes_masks = []
    height, width, _ = image.shape
    
    for box in boxes:
        mask = np.zeros((height, width), 'uint8')
        xmin, ymin, xmax, ymax = box[0],box[1],box[2],box[3]
        
        # fill polygon
        poly = np.array((
            (xmin,ymin),
            (xmax,ymin),
            (xmax,ymax),
            (xmin,ymax)
        ))
        
        rr, cc = polygon(poly[:,1], poly[:,0], (height, width))
        mask[rr,cc] = 1
        boxes_masks.append(mask)

    return boxes_masks

def mask_multilabel_classification (image, gt_boxes, gt_labels, pred_boxes, pred_labels, scores, n_classes, approach, th=0.5):
    """
    In addition to the local detection, a global image-level diagnosis
    score is proposed by applying an aggregation mechanism.
    The goal of this step is therefore to aggregate, for
    every image, the scores of the detected bounding boxes into
    a vector, providing a global image classification.
    
    SUMMARY --> Make local detections a multi-label vector. SEE 3.2.3.2 in TFM_MariaPostigo.pdf

    Parameters
    ----------
    image : image from which we apply object detection
    gt_boxes : ground truth bounding boxes
    gt_labels : ground truth labels
    pred_boxes : predicted boxes
    pred_labels : predicted labels
    scores : confidence scores for the detections
    n_classes : number of possible classes
    approach : The aggregation method [max, area, log, mean, sum]
    th : IoU threshold. The default is 0.5.

    Returns
    -------
    boxes_masks_per_label : list with the different boxes belonging to each label
    gt_multiclass : the multi-label ground truth vector [0,0,1,1,0,0,0]
    pred_multiclass : The multi-label predicted vector [0,0,1,0,1,0,0]
    unique_pred_labels : the predicted classes in the image
    """

    boxes_masks_per_label = []
    height, width, ch = image.shape #is a tensor
    gt_multiclass = np.zeros((n_classes,))
    pred_multiclass = np.zeros((n_classes,))
    unique_pred_labels = []
    
    indexes = np.argsort(pred_labels)
    pred_labels_sorted = pred_labels[indexes]
    pred_boxes_sorted = pred_boxes[indexes, :]
    scores_sorted = scores[indexes]
    
        
    for label in gt_labels:
        gt_multiclass[label-1]=1
    
    for oneClass in range(n_classes):
        scores_per_class = []
        max_score_overlap = 0
        area = 0
        
        #Get the indexes of a single label
        indexesOneClass = np.where(pred_labels_sorted==oneClass+1)
        pred_boxesOneClass = pred_boxes_sorted[indexesOneClass, :][0]
        scoresOneClass = scores_sorted[indexesOneClass]
        
        #2.Get the intexes of those scores >th
        indexesThreshold = np.where(np.array(scoresOneClass)>=th)[0]
        mask = np.zeros((height, width), dtype='float')
        
        if len(indexesThreshold)!=0:

            pred_boxesOneClass = pred_boxesOneClass[indexesThreshold].tolist()
            scoresOneClass = scoresOneClass[indexesThreshold]
        
        for box, score in zip(pred_boxesOneClass, scoresOneClass):
            
            scores_per_class.append(score)
            xmin, ymin, xmax, ymax = box[0],box[1],box[2],box[3]
    
            # fill polygon
            poly = np.array((
                (xmin,ymin),
                (xmax,ymin),
                (xmax,ymax),
                (xmin,ymax)
            ))
            
            rr, cc = polygon(poly[:,1], poly[:,0], (height, width))
            
            
            if (approach=='area'):
                area += ((xmax-xmin)/width)*((ymax-ymin)/height)*score
                mask[rr,cc] = float(score)
                
                            
            if approach == 'log':
                mask[rr,cc] = float(score)
            
            if approach == 'mean':
                mask[rr,cc] = float(score)
                
            if (approach=='max'):
                max_score_overlap = np.amax(mask[rr,cc])
                
                if(max_score_overlap>float(score)):#do the overlapping
                    mask[rr,cc] = np.where(mask[rr,cc] == 0, float(score), mask[rr,cc])
                    
                else:
                    mask[rr,cc] = float(score)
 
            if(approach=='sum'):
                mask[rr,cc] += float(score)
        
        if np.amax(mask)>0:
            boxes_masks_per_label.append(mask) 
            pred_multiclass[oneClass] = np.amax(mask)
            unique_pred_labels.append(oneClass+1)
            
        if(approach=='mean'):
            if(np.asarray(scores_per_class).size !=0):
                pred_multiclass[oneClass] = np.mean(scores_per_class) 
                # unique_pred_labels.append(oneClass+1)
                
        if (approach=='log'):
            if(np.asarray(scores_per_class).size !=0):
                pred_multiclass[oneClass] = logsumexp(scores_per_class)
                # unique_pred_labels.append(oneClass+1)
            
        if(approach=='area' or approach=='maxArea'):
            pred_multiclass[oneClass]=area

    return boxes_masks_per_label, gt_multiclass, pred_multiclass, unique_pred_labels


def _get_boxes_from_masks (boxes_masks):    
    """
    Given a set of masks, their corresponding bounding boxes are created
    Args: 
        boxes_masks: list of masks to convert into boxes
    """
    boxes = []
    for mask in boxes_masks:
        lb = label (mask)
        props = regionprops(lb)
        for prop in props:
          b = [prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]]
          boxes.append(b)

    return torch.as_tensor(boxes, dtype=torch.float32)

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(m.floor(angle / (m.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else m.pi - angle
    alpha = (sign_alpha % m.pi + m.pi) % m.pi

    bb_w = w * m.cos(alpha) + h * m.sin(alpha)
    bb_h = w * m.sin(alpha) + h * m.cos(alpha)

    gamma = m.atan2(bb_w, bb_w) if (w < h) else m.atan2(bb_w, bb_w)

    delta = m.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * m.cos(alpha)
    a = d * m.sin(alpha) / m.sin(delta)

    y = a * m.cos(gamma)
    x = y * m.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def applyRotation (img, mask, boxes_masks, theta):
  # Imagen
  img_array=np.array(img)
  # Máscara
  mask_array=np.array(mask)
  img_aux=rotate_image(img_array,theta)
  bb=largest_rotated_rect(img_aux.shape[1],img_aux.shape[0],theta)
  img_aux=crop_around_center(img_aux,bb[0],bb[1])

  mask_aux=rotate_image(mask_array,theta)
  bb=largest_rotated_rect(mask_aux.shape[1],mask_aux.shape[0],theta)
  mask_aux=crop_around_center(mask_aux,bb[0],bb[1]) 

  boxes_masks_aux = []
  for bm in boxes_masks:
    bm_array=np.array(bm)
    bm_aux=rotate_image(bm_array,theta)
    bb=largest_rotated_rect(bm_aux.shape[1],mask_aux.shape[0],theta)
    bm_aux=crop_around_center(bm_aux,bb[0],bb[1]) 
    boxes_masks_aux.append(bm_aux)

  # Comprobamos que no desaparezca el objeto ni toque los bordes, lo queremos tener al completo
  valid=True
  for bm_aux in boxes_masks_aux:
    borders=np.sum(bm_aux[:,0])+np.sum(bm_aux[:,-1])+np.sum(bm_aux[0,:])+np.sum(bm_aux[-1,:])
    if (borders==0 and np.sum(bm_aux)>0):
        pass
    else:
        valid=False
        return valid, [],[],[]

  borders=np.sum(mask_aux[:,0])+np.sum(mask_aux[:,-1])+np.sum(mask_aux[0,:])+np.sum(mask_aux[-1,:])
  if (borders==0 and np.sum(mask_aux)>0):
      pass
  else:
      valid=False
      return valid,[],[],[]
  if (valid):
      img=img_aux
      mask=mask_aux

  return valid, img, mask, boxes_masks_aux

