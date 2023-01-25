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
from google.colab.patches import cv2_imshow
from utils import ToTensor
# def pil_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

def cv2_loader(img_path, lbl_path):
  img = cv2.imread(img_path)[:, :, ::-1].copy()
  # 0 -> grey
  label = cv2.imread(lbl_path, 0)
  return img, label

class Cityscapes(VisionDataset):
    # togliere target transform (probabilmente domani adaptation)
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Cityscapes, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        
        with open('/content/drive/MyDrive/data/data/Cityscapes/info.json') as f:
          data = json.load(f)
          self.map_label = dict(data['label2train'])
          #self.mean, self.std  = tuple(data['mean']), tuple(data['std'])
          self.mean = [0.5, 0.5, 0.5]
          self.std = [0.5, 0.5, 0.5]
          self.to_tensor = ToTensor(self.mean, self.std)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        file = root + '/' + self.split + '.txt'
        self.l = []
        with open(file,'r') as f:
            lines = f.readlines()
            for line in lines:
              #take the image name
              #print(line)
              img_name  = line.split('/')[1][:-len('_leftImg8bit.png')-1]
              img = root + '/images/' +img_name.strip() + '_leftImg8bit.png'
              label = root + '/labels/' +img_name.strip() + '_gtFine_labelIds.png'
              #print(img,label)
              self.l.append((img,label))
    
    def __getitem__(self, index):
        path_image, path_label = self.l[index]
        image, label = cv2_loader(path_image, path_label)

        #label = self.map_label[label]
        label = np.vectorize(self.map_label.get)(label)
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        #print(f"before transform: {image.shape}, {label.shape}")
        # cv2_imshow(image)
        # cv2_imshow(label)
        if self.transform is not None:
            #print("Calling transform...")
            image, label = self.transform(image, label)
        # decidere se applicare sempre la to tensor
        #print(f"after transform: {image.shape}, {label.shape}")
        return self.to_tensor(image, label)

    def __len__(self):
        length = len(self.l) # Provide a way to get the length (number of elements) of the dataset
        return length