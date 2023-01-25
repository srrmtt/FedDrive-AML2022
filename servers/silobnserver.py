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
    """
    Instaciate an instance of the SiloBNServer 
    @params
      - clients: list, client list 
      - path: path where the server dump is stored, load this model if the path is not None
      - test_dataloader: DataLoader, contains the test images and labels
      - bn_layer: bool, true to averaging everything except the bn parameters, false exclude only the running ones
      - clients_ckpt_path: str, folder path where store and load the clients dumps
    """
    super().__init__(clients, None, test_dataloader, False)
    self.bn_layer = bn_layer
    self.updates = []

    if path:
      # load server model from path
      if os.path.isfile(path):
        print("Loading model from", path)
        checkpoint = torch.load(path)
        print(checkpoint['round'], checkpoint['mIoU_for_rounds'])
        self.main_model.load_state_dict(checkpoint["model_state_dict"])
        print("[SiloBNServer]: Model loaded correctly from dump")
      else:
        print(f"[SiloBNServer]:ERROR {path} is not a valid path.")

    if clients_ckpt_path:
      # load clients if a path is specified
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
    """
    Load server parameters of server on the specified client 
    """
    # client.bisenet_model.load_state_dict(client.bn_dict, strict=False)

    for k, v in self.main_model.state_dict().items():
      if self.bn_layer:
        # exclude bn params
        if 'bn' not in k:
          client.bisenet_model.state_dict()[k].data.copy_(v)
        else:
          # exclude bn running statistics
          if 'bn.running' not in k and 'bn.num_batches_tracked' not in k:
            client.model.state_dict()[k].data.copy_(v)
  
  def _aggregation(self):
    """
    Aggregate clients models to create an average model. This method will be used to update the server model.
    """
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
    """
    Sample some clients (specified in the NUM_CLIENTS_FOR_ROUND) and train them, then add their results in the updates
    list in this way they will be averaged with the aggregation method. 
    @return 
      - the train losses of the clients
    """
    clients = random.sample(self.clients,NUM_CLIENTS_FOR_ROUND)
    losses = dict()

    for i, c in enumerate(clients):
      self.load_server_model_on_client(c)
      num_samples, update, loss = c.train()
      losses[c.client_id] = loss
      self.updates.append((num_samples, update))

    return losses

  
  def update_model(self):
    """
    Load the averaged solution (updates) computed with the aggregation method and load it on the server.  
    """
    averaged_sol_n = self._aggregation()

    self.main_model.load_state_dict(averaged_sol_n, strict=False)
    self.updates = []
  
  def copy_bn_stats(self):
    bn_dict_tmp = OrderedDict()
    for k, v in self.main_model.state_dict().items():
        if 'bn' in k:
            bn_dict_tmp[k] = deepcopy(v)
    return bn_dict_tmp

  def evaluate(self):
    """
    Evaluate the server performance on the test set.
    """
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
    """
    Save clients in the given path directory
    """ 
    for client in self.clients:
      client.save_bn_stats(ckpt_path)
    print("[SiloBNServer]: Clients saved correctly.")

  @staticmethod
  def reset_bn_layers(net: nn.Module):
    """
    Reset running stats in the batch normalization layer of the given net
    """
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
          m.reset_running_stats()
  
  @staticmethod
  def compute_running_stats(net: nn.Module, loader: DataLoader):
    """
    Run the net in the forward mode to learn the running statistics of the test target. Torch no grad guarantees 
    that the net will not learn anything.
    """
    net.train()
    for images, _ in loader:
      with torch.no_grad():
        images = images.half().to(DEVICE)
        net(images)
        
  @staticmethod
  def print_running_stats(net: nn.Module):
    """
    Print the running statistics of the module. 
    """
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
          print(m.running_mean, m.running_var)





  
