from servers import Server, NUM_CLIENTS_FOR_ROUND
from torch.utils.data import DataLoader
from clients import Client, SiloBNClient
from typing import OrderedDict
from tqdm import tqdm
import random
import torch
from copy import deepcopy
import torch.nn as nn
import os
from utils import compute_mIoU, numpy_mean_iou

DEVICE = "cuda"
class SiloBNServer(Server):

  def __init__(self, clients: list, path: str, test_dataloader: DataLoader, bn_layer:bool = False, clients_ckpt_path:str=None):

    super().__init__(clients, None, test_dataloader, False)
    self.bn_layer = bn_layer
    self.updates = []

    if path:
      if os.path.isfile(path):
        print("Loading model from", path)
        checkpoint = torch.load(path)
        print(checkpoint['round'], checkpoint['mIoU_for_rounds'])
        self.main_model.load_state_dict(checkpoint["model_state_dict"])
        print("[SiloBNServer]: Model loaded correctly from dump")
      else:
        print(f"[SiloBNServer]:ERROR {path} is not a valid path.")

    if clients_ckpt_path:
      if os.path.isdir(clients_ckpt_path):
        for client in self.clients:
          _path = SiloBNClient.build_ckpt_path(clients_ckpt_path, client.client_id)
          if os.path.isfile(_path):
            checkpoint = torch.load(_path)
            client.bisenet_model.load_state_dict(checkpoint)
          else:
            print(f"[SiloBNServer]:ERROR {_path} is not a valid path.")

        print("[SiloBNServer]: Clients loaded correctly")

  def load_server_model_on_client(self, client: Client):
    # client.bisenet_model.load_state_dict(client.bn_dict, strict=False)

    for k, v in self.main_model.state_dict().items():
      if self.bn_layer:
        if 'bn' not in k:
          client.bisenet_model.state_dict()[k].data.copy_(v)
        else:
          if 'bn.running' not in k and 'bn.num_batches_tracked' not in k:
            client.model.state_dict()[k].data.copy_(v)
  
  def _aggregation(self):
    total_weight = 0.
    base = OrderedDict()

    for (client_samples, client_model) in self.updates:

        total_weight += client_samples
        for key, value in client_model.items():
          
          if self.bn_layer:
              if 'bn' not in key:
                  if key in base:
                      base[key] += client_samples * value.type(torch.FloatTensor)
                  else:
                      base[key] = client_samples * value.type(torch.FloatTensor)
          else:
            # ignore 'running mean' and the 'running variance' values
            if 'bn.running' not in key and 'bn.num_batches_tracked' not in key:
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
    averaged_sol_n = deepcopy(self.main_model.state_dict())
    for key, value in base.items():
        if total_weight != 0:
            averaged_sol_n[key] = value.to('cuda') / total_weight
    return averaged_sol_n

  def train(self):
    clients = random.sample(self.clients,NUM_CLIENTS_FOR_ROUND)
    losses = dict()

    for i, c in enumerate(clients):
      self.load_server_model_on_client(c)
      num_samples, update, loss = c.train()
      losses[c.client_id] = loss
      self.updates.append((num_samples, update))

    return losses

  
  def update_model(self):
    averaged_sol_n = self._aggregation()

    self.main_model.load_state_dict(averaged_sol_n, strict=False)
    self.updates = []

  def test_clients(self, clients_to_test: list):
    loss_test = dict()
    # save previous bn parameters 
    bn_dict_tmp = self.copy_bn_stats()
    self.reset_bn_layers()
    for client in clients_to_test:
      
      # use server 
      self.load_server_model_on_client(client)
      loss = client.test()
      loss_test[client.client_id] = loss
      self.model.load_state_dict(bn_dict_tmp, strict=False)
    return loss_test
  
  def copy_bn_stats(self):
    bn_dict_tmp = OrderedDict()
    for k, v in self.main_model.state_dict().items():
        if 'bn' in k:
            bn_dict_tmp[k] = deepcopy(v)
    return bn_dict_tmp

  def evaluate(self):
    # save previous bn parameters 
    bn_dict_tmp = self.copy_bn_stats()

    net = self.main_model.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    
    net=net.half()
    # compute bn running stats on the evaluation set
    SiloBNServer.reset_bn_layers(net)
    SiloBNServer.compute_running_stats(net, self.test_dataloader)
    net.train(False) # Set Network to evaluation mode
    running_corrects = 0
    
    torch.cuda.empty_cache() 
    mIoU = 0
    count = 0
    
    for images, labels in self.test_dataloader:
      images = images.half().to(DEVICE)
      labels = labels.half().to(DEVICE)
      # Forward Pass
      outputs = net(images,test=True,use_test_resize=False)
      preds = outputs.argmax(dim=1)
      mIoU += compute_mIoU(labels,preds)
      count += 1    

    print("\nmIoU = ",mIoU/count)
    net.train(True)
    net.load_state_dict(bn_dict_tmp, strict=False)
    return mIoU/count
  
  def clients_dump(self, ckpt_path: str):
    for client in self.clients:
      client.save_bn_stats(ckpt_path)
    print("[SiloBNServer]: Clients saved correctly.")

  @staticmethod
  def reset_bn_layers(net: nn.Module):
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
          m.reset_running_stats()
  
  @staticmethod
  def compute_running_stats(net: nn.Module, loader: DataLoader):
    net.train()
    for images, _ in loader:
      with torch.no_grad():
        images = images.half().to(DEVICE)
        net(images)
        
  @staticmethod
  def print_running_stats(net: nn.Module):
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
          print(m.running_mean, m.running_var)





  
