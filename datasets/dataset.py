import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import os.path
import json
import numpy as np
import cv2
from utils import ToTensor

def cv2_loader(img_path: str, lbl_path: str) -> tuple:
  """
  Load an image and its label as a numpy array from their path. 
  """
  img = cv2.imread(img_path)[:, :, ::-1].copy()
  label = cv2.imread(lbl_path, 0)
  return img, label

JSON_PATH = '/content/drive/MyDrive/data/data/Cityscapes/info.json'
class Cityscapes(VisionDataset):
  def __init__(self, root: str, split: str='train', transform:list=None, target_transform=None):
    """ 
    Return an instance of the whole Cityscape dataset.
    @params:
      - root: str, path of the directory that contains the train and test txt file
      - split: str, boolean meaning: training for loading the training set images, otherwise test set
      - transform: list, optional parameter containing the possible transformations
    """
    super(Cityscapes, self).__init__(root, transform=transform, target_transform=target_transform)
    # assign image transformation
    self.transform = transform
    
    # take json file parameter 
    with open(JSON_PATH) as f:
      data = json.load(f)
      self.map_label = dict(data['label2train'])
      self.mean = [0.5, 0.5, 0.5]
      self.std = [0.5, 0.5, 0.5]
      self.to_tensor = ToTensor(self.mean, self.std)
    # This defines the split you are going to use
    # (split files are called 'train.txt' and 'test.txt')
    self.split = split 

    file = root + '/' + self.split + '.txt'
    self.l = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
          #take the image name
          img_name  = line.split('/')[1][:-len('_leftImg8bit.png')-1]
          img = root + '/images/' +img_name.strip() + '_leftImg8bit.png'
          label = root + '/labels/' +img_name.strip() + '_gtFine_labelIds.png'
          #print(img,label)
          self.l.append((img,label))
  
  def __getitem__(self, index: int) -> tuple:
    """
    Return the image and the label from the dataset.
    """
    path_image, path_label = self.l[index]
    image, label = cv2_loader(path_image, path_label)

    # map label from the JSON 
    label = np.vectorize(self.map_label.get)(label)

    # Applies preprocessing when accessing the image if any transformations are specified
    if self.transform is not None:
        image, label = self.transform(image, label)

    return self.to_tensor(image, label)

  def __len__(self) -> int :
    """
    Provide a way to get the length (number of elements) of the dataset
    """
    length = len(self.l) 
    return length