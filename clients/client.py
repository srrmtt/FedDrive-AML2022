# -*- coding: utf-8 -*-

import torchvision
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, Subset, DataLoader
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


NUM_CLASSES = 19      # Dataset Classes

BATCH_SIZE = 4        # This parameter can't be changed due to poor memory 
LR = 5*1e-3           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default -5

NUM_EPOCHS = 2
STEP_SIZE = 20       #20 How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

DEVICE = 'cuda'

scales = (0.25, 2.)
cropsize = (512, 1024)
# taken from paper
train_transformations = [RandomResizedCrop(scales, cropsize), RandomHorizontalFlip(
), ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)]


JSON_PATH = '/content/drive/MyDrive/data/data/Cityscapes/info.json'


def cv2_loader(img_path, lbl_path):
    img = cv2.imread(img_path)[:, :, ::-1].copy()
    # 0 -> grey
    label = cv2.imread(lbl_path, 0)
    return img, label


class DatasetClient(VisionDataset):
  def __init__(self, root: str, lines: list, transform=None, target_transform=None):
    super(DatasetClient, self).__init__(
        root, transform=transform, target_transform=target_transform)
    self.transform = transform
    self.l = []
    with open(JSON_PATH) as f:
        data = json.load(f)
        self.map_label = dict(data['label2train'])
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.to_tensor = ToTensor(self.mean, self.std)
        self.root = root
        for line in lines:
            # take the image name
            img_name = line.split('/')[1][:-len('_leftImg8bit.png')-1]
            img = root + '/images/' + img_name.strip() + '_leftImg8bit.png'
            label = root + '/labels/' + img_name.strip() + '_gtFine_labelIds.png'
            # append both path to the list
            self.l.append((img, label))

  def __getitem__(self, index: int) -> tuple:
    """
    Return the image and label corresponding to the given index in a tensor format. 
    """
    # get label path and image path from list
    path_image, path_label = self.l[index]
    image, label = cv2_loader(path_image, path_label)
    # apply the mapping from grey to label (taken from JSON)
    label = np.vectorize(self.map_label.get)(label)
    # Applies preprocessing when accessing the image
    if self.transform is not None:
        image, label = self.transform(image, label)
    return self.to_tensor(image, label)

  def __len__(self):
      """
      Provide a way to get the length (number of elements) of the dataset
      """
      return len(self.l)


class Client():
    def __init__(self, id_, dataset, pseudo_lab=False, teacher_model=None, scheduler:bool = False):
        self.client_id = id_
        # training dataset
        self.dataset = dataset
        # train transformations
        self.train_transformations = Compose(train_transformations)
        # train loader
        self.dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
        # main model
        self.bisenet_model = BiSeNetV2(
            n_classes=NUM_CLASSES, output_aux=True, pretrained=True)
        self.bisenet_model.requires_grad = True
        # ignore index 255 when computing the cross entropy since is a 'unlabeled' label
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=255, reduction='none')  
        self.parameters_to_optimize = self.bisenet_model.parameters()
        self.optimizer = optim.SGD(
            self.parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if scheduler:
          self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        # style augment instance
        self.style_detector = StyleAugment(20, L=0.1,b=1)
        self.avg_style = None

        # pseudo-labels
        self.pseudo_lab = pseudo_lab
        if self.pseudo_lab:
            self.criterion = SelfTrainingLoss()
            self.criterion.set_teacher(teacher_model)

    def compute_avg_style(self) -> np.array:
      """
      This method compute the avarage style from the dataset contained in the client instance, this features is provided
      by the StyleAugment instance (style_detector) in the client object. Exploiting this, the avg style will be set in the 
      first position of the styles list into the style detector object. See style.py file to have more information. 
      @return
        - a numpy array with the average style
      """
      self.style_detector.add_style(self.dataset)
      self.avg_style = self.style_detector.styles[0]
      return self.avg_style

    def generate_update(self):
      """
      This method return a copy of the state dict of the client model.
      """
      return deepcopy(self.bisenet_model.state_dict())

    def train(self) -> tuple:
      """
      This method train the client net for the number of epochs specified in the costant NUM_EPOCHS, if the client uses
      pseudo labels it will train with them using the criterion specified during the instantiation, otherwise the train 
      follows the classic train on the bisenet network.
      @return
        - number of training samples: int
        - updated model: dict, it's the updated net state model 
        - training loss: list

      """
      num_train_samples = len(self.dataset)

      # this will bring the network to GPU if DEVICE is cuda
      net = self.bisenet_model.half().to(DEVICE)

      cudnn.benchmark  # Calling this optimizes runtime
      # training loss list 
      loss_t = []  

      for epoch in range(NUM_EPOCHS):
          current_step = 0

          loss_e = []
          # Iterate over the dataset
          for images, labels in self.dataloader:
              # Bring data over the device of choice
              images = images.half().to(DEVICE)
              labels = labels.half().to(DEVICE)
              
              # Sets module in training mode
              net.train()  

              # PyTorch, by default, accumulates gradients after each backward pass
              # We need to manually set the gradients to zero before starting a new iteration
              self.optimizer.zero_grad()  

              # Forward pass to the network
              output1, output2, output3, output4, output5 = net(images)

              if self.pseudo_lab == False:
                  # standard label case, labels are provided by manual pixel annotations
                  loss1 = self.criterion(output1, labels.long())[
                      labels != 255].mean()
                  loss2 = self.criterion(output2, labels.long())[
                      labels != 255].mean()
                  loss3 = self.criterion(output3, labels.long())[
                      labels != 255].mean()
                  loss4 = self.criterion(output4, labels.long())[
                      labels != 255].mean()
                  loss5 = self.criterion(output5, labels.long())[
                      labels != 255].mean()
              else:
                  # training with pseudo labels
                  loss1 = self.criterion(output1, images).mean()
                  loss2 = self.criterion(output2, images).mean()
                  loss3 = self.criterion(output3, images).mean()
                  loss4 = self.criterion(output4, images).mean()
                  loss5 = self.criterion(output5, images).mean()
              loss = loss1+loss2+loss3+loss4+loss5
              loss_e.append(loss.item())

              # loss backpropagation
              loss.backward()

              # update weights based on accumulated gradients
              self.optimizer.step()  

              current_step += 1
          # average loss with batch size
          loss_t.append(sum(loss_e)/BATCH_SIZE)
          if self.scheduler:
            self.scheduler.step()
          # get the copy of the client net state dict 
          update = self.generate_update()
      return num_train_samples, update, loss_t
