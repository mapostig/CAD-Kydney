import random
from bbox_utils import _get_masks_from_boxes, _get_boxes_from_masks
import numpy as np
import torch
import cv2
import skimage
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

def applyRotation2 (img, mask, mask_boxes, angle):
  img_rot = rotate(img, angle)
  mask_rot = rotate(mask, angle)
  boxes_masks_rot = []
  for i in range(len(mask_boxes)):
    rot_box = rotate(mask_boxes[i], angle)
    boxes_masks_rot.append(rot_box)

  return img_rot, mask_rot, boxes_masks_rot

class RotationTransform (object):
    """Rotate by 
        angle: 56*random.random()-28 """

    def __init__(self):
        degrees = 56*random.random()-28
        self.degrees = (-degrees, degrees)

    def __call__(self, img, target):
        
        angle = random.uniform(self.degrees[0], self.degrees[1])
#        angle = 3.8
        mask = target['mask']

        #get masks from boxes
        mask_boxes = _get_masks_from_boxes(img, target['boxes'])

        img_rot, mask_rot, boxes_masks_rot = applyRotation2 (img, mask, mask_boxes, angle)

        boxes_masks = []
        for bmask in boxes_masks_rot:
          bmask = np.where (bmask!=0, 1, bmask)
          boxes_masks.append(bmask)


        target['boxes'] =  _get_boxes_from_masks (boxes_masks)
        target['mask'] = mask_rot
        return img_rot, target

class MaskTransform (object):

   """Crop the image using the lesion mask.
   Args:
       border (tuple or int): Border surrounding the mask. We dilate the mask 
   """
   def __init__(self, border):
       assert isinstance(border, (int, tuple))
       if isinstance(border, int):
           self.border = (border,border)
       else:
           self.border = border

   def __call__(self, image, target):
       
       mask = target['mask']
       boxes = target['boxes']
       h, w = image.shape[:2]
       #Calculamos los índices del bounding box para hacer el cropping
       sidx=np.nonzero(mask)
       minx=np.maximum(sidx[1].min()-self.border[1],0)
       maxx=np.minimum(sidx[1].max()+1+self.border[1],w)
       miny=np.maximum(sidx[0].min()-self.border[0],0)
       maxy=np.minimum(sidx[0].max()+1+self.border[1],h)
       #Recortamos la imagen
       crop_img=image[miny:maxy,minx:maxx,...]
       crop_mask=mask[miny:maxy,minx:maxx]

       #Reescribimos las coordenadas de la bbox dado el nuevo tamaño de la imagen 
       # print(miny,maxy,minx,maxx)
       new_boxes = boxes.clone()
       if len(boxes)!=0:
                  
         new_boxes[:, 0] = boxes[:, 0] - minx
         w = boxes[:,2]-boxes[:,0]
         new_boxes[:, 2] = new_boxes[:, 0] + w

         new_boxes[:, 1] = boxes[:, 1] - miny
         h = boxes[:,3]-boxes[:,1]
         new_boxes[:, 3] = new_boxes[:, 1] + h
       
       target['mask']=crop_mask
       target['boxes']=new_boxes

       return crop_img, target

def myflip(image, boxes):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = np.fliplr(image)

    # Flip boxes
    if len(boxes)==0:
      new_boxes = boxes.clone()
    else:
      
      H, W, ch = image.shape
      for box in boxes: 
        xmax = box[2]
        w = xmax - box[0]
        box[0]=W-(w+box[0])
        box[2]=box[0]+w

    return new_image, boxes

class HorizontalFlipTransform(object):
    
    """Horizontally flip the given numpy ndarray randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, image, target):
        """
        Args:
            sample (dict):  image, mask, boxes, labels

        Returns:
            ndarray: Randomly flipped image, mask and boxes
        """
#        mask = sample['mask']
        boxes = target['boxes']
        mask = target['mask']

        if random.random() < self.p:
            image, boxes = myflip(image, boxes)
            mask = np.fliplr(mask)
            target['boxes']=boxes
            target['mask']=mask
        
        return image, target
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, target):

      # swap color axis because
      # numpy image: H x W x C
      # torch image: C X H X W
      mask = target['mask']
      mask = torch.from_numpy(np.flip(mask,axis=0).copy())
      
      target['mask']=mask

      image = image.transpose((2, 0, 1))
      return torch.from_numpy(np.flip(image,axis=0).copy()), target
      # return torch.as_tensor(image.transpose((2,0,1)).copy(), dtype=torch.float), target

# class ResizeTransform(object):
   
#    def __init__(self, output_size):
#        assert isinstance(output_size, (int, tuple))
#        self.output_size = output_size

#    def __call__(self, image, target):
       
#        boxes = target['boxes']

#        h, w = image.shape[:2]
#        if isinstance(self.output_size, int):
#            if h > w:
#                new_h, new_w = self.output_size * h / w, self.output_size
#            else:
#                new_h, new_w = self.output_size, self.output_size * w / h
#        else:
#            new_h, new_w = self.output_size

#        new_h, new_w = int(new_h), int(new_w)

#        new_image = transform.resize(image, (new_h, new_w))
# #        new_mask = transform.resize(mask, (new_h, new_w))
   
#        # Resize bounding boxes
#        # print('Previous',boxes)
#        if len(boxes)!=0:

#          x_scale = self.output_size[1] / image.shape[1]
#          y_scale = self.output_size[0] / image.shape[0]
#          # print(x_scale, y_scale)
         
#          boxes[:,0] = int(np.round(boxes[:,0] * x_scale))
#          boxes[:,1] = int(np.round(boxes[:,1] * y_scale))
#          boxes[:,2] = int(np.round(boxes[:,2] * x_scale))
#          boxes[:,3] = int(np.round(boxes[:,3] * y_scale))
#          # print('new box', boxes)
#          target['boxes']=boxes
         
#        return image, target
"""  
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):#realmente normalizaremos en la red

        img = image.type(torch.FloatTensor) 
#        mask = mask
#         mean = torch.as_tensor(self.mean[:,np.newaxis], dtype=dtype, device=img.device)
#         std = torch.as_tensor(self.std[:,np.newaxis], dtype=dtype, device=img.device)
        # img = (img - mean[:, None]) / std[:, None]
        img = img/255
        return  img, target
"""
class Normalize(object):
    """Normalize data by subtracting means and dividing by standard deviations.

    Args:
        mean_vec: Vector with means. 
        std_vec: Vector with standard deviations.
    """

    def __init__(self, mean,std):
      
        assert len(mean)==len(std),'Length of mean and std vectors is not the same'
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self,  image, target):
        
        c, h, w = image.shape
        assert c==len(self.mean), 'Length of mean and image is not the same' 
        image = image.type(torch.FloatTensor)
        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        mean = torch.squeeze(mean)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
    
        
        return image, target
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomIncreaseBrightness (object):
    
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img, target):

        if random.random() < self.p:
            alpha = random.choice(np.linspace(0.5,2,16))
            beta = random.choice(np.linspace(0,50,51))
            img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype),0, beta)
            
        return img, target
    

class AddNoise(object):

    def __call__(self, img, target):
      """
        Adds Gaussian Noise to the image.
      """
      a = np.random.binomial(1, 0.2, 1)

      if a==1:
        h, w , ch = img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(h,w,ch))
        gauss = gauss.reshape(h,w,ch)
        noisy = img + gauss
      
      else:
        noisy = img

      return noisy, target