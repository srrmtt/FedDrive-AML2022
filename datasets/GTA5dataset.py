import torchvision
import random
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import os.path
import sys
import json
import numpy as np
import cv2
from utils import ToTensor, StyleAugment


def cv2_loader(img_path, lbl_path):
    img = cv2.imread(img_path)[:, :, ::-1].copy()
   
    label = cv2.imread(lbl_path)
    return img, label

class GTA5(VisionDataset):
    def __init__(self, root: str, transform=None,fda=False,styles=None):
        super(GTA5, self).__init__(root, transform, target_transform=None)
        self.transform = transform
        # label and images root reading
        TRAIN_FILE_PATH = root + "/train.txt"
        LABEL_PATH = root + "/labels/"
        IMAGE_PATH = root + "/images/"
        JSON_PATH =  root + "/info.json"
        self.paths = []
        self.fda=fda
        if self.fda:
          self.current_style=None
          self.style_detector = StyleAugment(20)
          self.styles=styles


        
        with open(TRAIN_FILE_PATH, 'r') as f_in:
            for line in f_in:
                image = IMAGE_PATH + line.strip()
                label = LABEL_PATH + line.strip()
                self.paths.append((image, label))
        # json reading
        with open(JSON_PATH, 'r') as f_json:
            data = json.load(f_json)
            self.map_label = dict(data['label2train'])
            palette = [tuple(p[::-1]) for p in data['palette']]
            self.palette = dict(zip(palette, list(range(19))))
            self.mean, self.std = tuple(data['mean']), tuple(data['std'])
            self.to_tensor = ToTensor(self.mean, self.std)
    def pixel_mapping(self,rgb_image: np.array):
        grey_img = []
        for row in rgb_image:
            mapped_row = []
            for pixel in row:
                
                key = tuple(pixel)
                grey_pxl = self.palette.get(key, 255) 
                mapped_row.append(grey_pxl)
            grey_img.append(mapped_row)
        return np.vstack(grey_img)  
    def matrix_mapping(self,img):
      # Get keys and values
      k = np.array(list(self.palette.keys()))
      v = np.array(list(self.palette.values()))

      # Setup scale array for dimensionality reduction
      s = 256**np.arange(3)

      # Reduce k to 1D
      k1D = k.dot(s)

      # Get sorted k1D and correspondingly re-arrange the values array
      sidx = k1D.argsort()
      k1Ds = k1D[sidx]
      vs = v[sidx]

      # Reduce image to 2D
      labelOld2D = np.tensordot(img, s, axes=((-1),(-1)))

      # Get the positions of 1D sorted keys and get the correspinding values by
      # indexing into re-arranged values array
      out = vs[np.searchsorted(k1Ds, labelOld2D)]
      return out
    def __getitem__(self, index):
        path_image, path_label = self.paths[index]
        image, label = cv2_loader(path_image, path_label)
        # mapping in grey scale
        # label = np.vectorize(self.map_label.get)(label)
        # apply transormation
        
        label = self.matrix_mapping(label)
        if self.fda:
           
           style_detector = random.choice(self.styles)
           
           #cv2.imwrite("./before.png",image)

           
           
           image = style_detector.apply_style(image)
           #cv2.imwrite("./after.png",image)

        if self.transform is not None:
            image, label = self.transform(image, label)
        # normalize and return label and image
        return self.to_tensor(image, label)
    
    def __len__(self):
        return len(self.paths)
        
                 
                
