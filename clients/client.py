# -*- coding: utf-8 -*-

import torchvision
from sklearn.metrics import confusion_matrix  
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import os.path
import sys
from torch.backends import cudnn
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from copy import deepcopy
from tqdm import tqdm

from datasets import Cityscapes
from models import BiSeNetV2
from utils import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Compose, StyleAugment
from utils import SelfTrainingLoss


NUM_CLASSES = 19 # 101 + 1: There is am extra Background class that should be removed 

BATCH_SIZE = 4     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results
LR = 5*1e-3           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default -5

NUM_EPOCHS =2      # 20/30 Total number of training epochs (iterations over dataset)
#STEP_SIZE = 20       #20 How many epochs before decreasing learning rate (if using a step-down policy)
#GAMMA = 0.1          # Multiplicative factor for learning rate step-down

#LOG_FREQUENCY = 10
DEVICE = 'cuda'

scales=(0.25, 2.)
cropsize=(512, 1024)
train_transformations = [RandomResizedCrop(scales,cropsize),RandomHorizontalFlip(),ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4)]

def cv2_loader(img_path, lbl_path):
  img = cv2.imread(img_path)[:, :, ::-1].copy()
  # 0 -> grey
  label = cv2.imread(lbl_path, 0)
  return img, label

def compute_mIoU(y_true, y_pred,n_classes):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.cpu().flatten()
     y_true = y_true.cpu().flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)


class DatasetClient(VisionDataset):
  def __init__(self,root,lines, transform=None,target_transform=None):
      super(DatasetClient, self).__init__(root, transform=transform, target_transform=target_transform)
      self.transform = transform
      self.l = []
      with open('/content/drive/MyDrive/data/data/Cityscapes/info.json') as f:
        data = json.load(f)
        self.map_label = dict(data['label2train'])
        #self.mean, self.std  = tuple(data['mean']), tuple(data['std'])
        self.mean = [0.5,0.5,0.5]
        self.std = [0.5,0.5,0.5]
        self.to_tensor = ToTensor(self.mean, self.std)
        self.root = root
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

class Client():
    def __init__(self,id_,dataset,pseudo_lab=False,teacher_model=None):
        self.client_id = id_
        self.dataset = dataset
        self.train_transformations = Compose(train_transformations)
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
        self.bisenet_model = BiSeNetV2(n_classes=NUM_CLASSES,output_aux=True,pretrained=True) 
        self.bisenet_model.requires_grad = True
        self.criterion = nn.CrossEntropyLoss(ignore_index=255,reduction='none') # da consegna ignore_index=255
        self.parameters_to_optimize = self.bisenet_model.parameters() 
        self.optimizer = optim.SGD(self.parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        # style augment instance 
        self.style_detector = StyleAugment(20,L=0.01)
        self.avg_style = None

        #pseudo-labels
        self.pseudo_lab=pseudo_lab
        if self.pseudo_lab:
          self.criterion = SelfTrainingLoss()
          self.criterion.set_teacher(teacher_model) 


    def compute_avg_style(self):
      self.style_detector.add_style(self.dataset)
      self.avg_style = self.style_detector.styles[0]
      return self.avg_style

    
    
    
    def generate_update(self):
      return deepcopy(self.bisenet_model.state_dict())
    
    
    def train(self):
        num_train_samples = len(self.dataset)

        net = self.bisenet_model.half().to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

        cudnn.benchmark # Calling this optimizes runtime

        loss_t = [] #new
        for epoch in range(NUM_EPOCHS):
          #print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, self.scheduler.get_lr()))
          current_step=0

          loss_e = []
          # Iterate over the dataset
          for images, labels in self.dataloader:
            # Bring data over the device of choice
            
            images = images.half().to(DEVICE)
            labels = labels.half().to(DEVICE)
            #print("GPU Allocation after moving images and labels")
            #!nvidia-smi
            net.train() # Sets module in training mode
        
            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            self.optimizer.zero_grad() # Zero-ing the gradients
        
            # Forward pass to the network
            
            output1,output2,output3,output4,output5 = net(images)
            if self.pseudo_lab==False:
              
              loss1 = self.criterion(output1,labels.long())[labels!=255].mean()
              loss2 = self.criterion(output2,labels.long())[labels!=255].mean()
              loss3 = self.criterion(output3,labels.long())[labels!=255].mean()
              loss4 = self.criterion(output4,labels.long())[labels!=255].mean()
              loss5 = self.criterion(output5,labels.long())[labels!=255].mean()
            else:
              #print("training with pseudo labels...")
              loss1 = self.criterion(output1,images).mean()
              loss2 = self.criterion(output2,images).mean()
              loss3 = self.criterion(output3,images).mean()
              loss4 = self.criterion(output4,images).mean()
              loss5 = self.criterion(output5,images).mean()
            loss = loss1+loss2+loss3+loss4+loss5
            # print("Loss:"+str(loss.item()))
            loss_e.append(loss.item())
            # Log loss
            #if current_step % LOG_FREQUENCY == 0:
              #print('Step {}, Loss {}'.format(current_step, loss.item()))
              #mIoU = compute_mIoU(labels,pred1,n_classes=19)
              #print('Step {}, mIoU {}'.format(current_step, mIoU))
        
            loss.backward()
            
            self.optimizer.step() # update weights based on accumulated gradients
        
            current_step += 1
            
          loss_t.append(sum(loss_e)/BATCH_SIZE)          
          # Step the scheduler
          #self.scheduler.step()
          
          update = self.generate_update()
        return num_train_samples, update,loss_t
         
        
        