# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:27:31 2022

@author: matte
"""
import random
from torch.backends import cudnn
import numpy as np
import torch
from models import BiSeNetV2
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
import torch.nn as nn
import torch.optim as optim



BATCH_SIZE = 4     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results
LR = 1*1e-3           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default -5

STEP_SIZE = 20       #20 How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10

DEVICE = "cuda"
NUM_CLASSES = 19
NUM_ROUNDS = 100
NUM_CLIENTS_FOR_ROUND = 5
NUM_EPOCHS = 600


def _fast_hist(n_classes, label_true, label_pred):
      mask = (label_true >= 0) & (label_true < n_classes)
      hist = np.bincount(
          n_classes * label_true[mask].astype(int) + label_pred[mask],
          minlength=n_classes ** 2,
      ).reshape(n_classes, n_classes)
      return hist


def compute_mIoU(y_true,y_pred):
  
  y_pred = y_pred.cpu().detach().numpy().flatten()
  y_true = y_true.cpu().detach().numpy().flatten()
  # print(y_true)
  # print(y_pred)
  # index = [i for i in range(len(y_true)) if y_true[i] != 255]
  # mask = y_true == 255
  # y_true = np.delete(y_true, np.where(mask))
  # y_pred = np.delete(y_pred, np.where(mask))
  #print(index)
  # print(y_true)
  # print(y_pred)
  #hist = confusion_matrix(y_true, y_pred, labels=range(19))
  hist = _fast_hist(19,y_true,y_pred)
  #print(hist)
  gt_sum = hist.sum(axis=1)
  mask = (gt_sum != 0)
  diag = np.diag(hist)
  iu = diag / (gt_sum + hist.sum(axis=0) - diag)
  mean_iu = np.mean(iu[mask])
  return mean_iu

# def compute_mIoU(mask,pred_mask,smooth=1e-10,n_classes=19):
#     with torch.no_grad():
#         #pred_mask = F.softmax(pred_mask, dim=1)
#         pred_mask = torch.argmax(pred_mask, dim=1)
#         pred_mask = pred_mask.contiguous().view(-1)
#         mask = mask.contiguous().view(-1)

#         iou_per_class = []
#         for clas in range(0, n_classes): #loop per pixel class
#             true_class = pred_mask == clas
#             true_label = mask == clas

#             if true_label.long().sum().item() == 0: #no exist label in this loop
#                 iou_per_class.append(np.nan)
#             else:
#                 intersect = torch.logical_and(true_class, true_label).sum().float().item()
#                 union = torch.logical_or(true_class, true_label).sum().float().item()

#                 iou = (intersect + smooth) / (union +smooth)
#                 iou_per_class.append(iou)
#         return np.nanmean(iou_per_class)

class Server_FDA():
    
    def __init__(self,clients,train_dataloader: DataLoader,test_dataloaderA: DataLoader,test_dataloaderB: DataLoader,styles = None):
        self.test_dataloaderA=test_dataloaderA
        self.test_dataloaderB=test_dataloaderB
        self.train_dataloader=train_dataloader
        self.clients = clients
        self.main_model = BiSeNetV2(n_classes=19,pretrained=True)
        self.styles = styles
        self.criterion = nn.CrossEntropyLoss(ignore_index=255,reduction='none') # da consegna ignore_index=255
        self.parameters_to_optimize = self.main_model.parameters() 
        self.optimizer = optim.SGD(self.parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=600, power=0.8, last_epoch=- 1, verbose=False)
        
    def load_server_model_on_client(self,client):
      client.bisenet_model.load_state_dict(deepcopy(self.main_model.state_dict()))

    
    def evaluate(self):
      net = self.main_model.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
      net.train(False) # Set Network to evaluation mode
      net=net.half()
      running_corrects = 0
      torch.cuda.empty_cache() 
      mIoUA = 0
      mIoUB = 0
      countA = 0
      countB = 0
      for images, labels in tqdm(self.test_dataloaderA):
        images = images.half().to(DEVICE)
        labels = labels.half().to(DEVICE)

        

        print("images:"+str(images.size()))
        print("labels:"+str(labels.size()))
        
        ##train_dataset.current_style=style
        # Forward Pass
        outputs = net(images,test=True,use_test_resize=False)
        preds = outputs.argmax(dim=1)
        # print(outputs.size())
        mIoUA += compute_mIoU(labels,preds)
        
        countA += 1

      for images, labels in tqdm(self.test_dataloaderB):
        images = images.half().to(DEVICE)
        labels = labels.half().to(DEVICE)

        

        print("images:"+str(images.size()))
        print("labels:"+str(labels.size()))
        
        ##train_dataset.current_style=style
        # Forward Pass
        outputs = net(images,test=True,use_test_resize=False)
        preds = outputs.argmax(dim=1)
        # print(outputs.size())
        mIoUB += compute_mIoU(labels,preds)
        
        countB += 1  
        
        # iou = ops.box_iou(labels, outputs)

        # print('IOU : ', iou.numpy()[0][0])
      print("mIoU_A = ",mIoUA/countA)
      print("mIoU_B = ",mIoUB/countB)
      net.train(True)
      return mIoUA/countA,mIoUB/countB

    def train(self):
      #num_train_samples = len(self.train_dataset)
      # check_epoch = 0
      # mIoUA = []
      # mIoUB = []

      checkpoint = torch.load('/content/drive/MyDrive/step4/checkpoints/long_esp_FDA/DatasetA/110checkpoint.pt')
      
      self.main_model.load_state_dict(checkpoint['model_state_dict'])
      if 'optimizer' in checkpoint:
        self.optimizer.load_state_dict(checkpoint['optimizer'])
      self.main_model.to(DEVICE)
      check_epoch = checkpoint['round']
      mIoUA = checkpoint['mIoUA']
      mIoUB = checkpoint['mIoUB']

      net = self.main_model.half().to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

      cudnn.benchmark # Calling this optimizes runtime


      for epoch in range(check_epoch,NUM_EPOCHS):
        print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, self.scheduler.get_lr()))
        current_step=0
        
        # Iterate over the dataset
        for images, labels in self.train_dataloader:
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
          
          
          loss1 = self.criterion(output1,labels.long())[labels!=255].mean()
          loss2 = self.criterion(output2,labels.long())[labels!=255].mean()
          loss3 = self.criterion(output3,labels.long())[labels!=255].mean()
          loss4 = self.criterion(output4,labels.long())[labels!=255].mean()
          loss5 = self.criterion(output5,labels.long())[labels!=255].mean()
      
          loss = loss1+loss2+loss3+loss4+loss5
          
          # Log loss
          if current_step % 10 == 0:
            print('Step {}, Loss {}'.format(current_step, loss.item()))
          #    mIoU = compute_mIoU(labels,pred1,n_classes=19)
          #    print('Step {}, mIoU {}'.format(current_step, mIoU))
      
          loss.backward()
          
          self.optimizer.step() # update weights based on accumulated gradients
      
          current_step += 1        
        # Step the scheduler
        self.scheduler.step()
        if epoch % 5 == 0:
          mioua,mioub=self.evaluate()
          mIoUB.append(mioub)
          mIoUA.append(mioua)
        if epoch % 10 == 0:
          torch.save({
              'round': epoch,
              'model_state_dict': self.main_model.state_dict(),
              'mIoUA' : mIoUA,
              'mIoUB':mIoUB,
              'optimizer': self.optimizer.state_dict}, '/content/drive/MyDrive/step4/checkpoints/long_esp_FDA/DatasetA/'+ str(epoch) +'checkpoint.pt')
        
            
        
        
            
        