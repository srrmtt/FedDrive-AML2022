from typing import OrderedDict

from clients import Client
from utils import compute_mIoU, numpy_mean_iou

import torch 
import os
from torch.utils.data import Dataset
from copy import deepcopy

DEVICE = "cuda"

class SiloBNClient(Client):
  def __init__(self, client_id: int, dataset: Dataset, pseudo_lab: bool=False, teacher_model = None,
    ckpt_path:str=None, name=None):

    super().__init__(client_id, dataset, pseudo_lab, teacher_model)
    self.bn_dict = OrderedDict()
    self.name = name
    self.ckpt_path = ckpt_path

    for k, v in self.bisenet_model.state_dict().items():
      if 'bn' in k:
        self.bn_dict[k] = deepcopy(v)

  def copy_bn_stats(self):
    """
    Return a copy of the batch normalization layer parameter of the bisenet model.
    """
    bn_dict_tmp = OrderedDict()
    for k, v in self.bisenet_model.state_dict().items():
        if 'bn' in k:
            bn_dict_tmp[k] = deepcopy(v)
    return bn_dict_tmp

  def save_bn_stats(self, ckpt_path: str):
    # this commented part could make the training more shallow but the performances are worst

    # for k, v in self.bisenet_model.state_dict().items():
    #     if 'bn' in k:
    #         self.bn_dict[k] = deepcopy(v)
    # save the client mdoel
    path = SiloBNClient.build_ckpt_path(str(self.client_id))
    torch.save(self.bisenet_model.state_dict(), path)
  
  @staticmethod
  def build_ckpt_path(client_id: int) -> str:
    """
    Given the root and the client id it returns the complete path of the client dump.
    """
    if self.ckpt_path:
      return os.path.join(self.ckpt_path, str(client_id) + "_bn.ckpt")
    print("Specify a checpoint path in the constructor.")




        

