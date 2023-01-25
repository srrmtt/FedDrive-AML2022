# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:27:31 2022

@author: matte
"""
import random
import numpy as np
import torch

from models import BiSeNetV2
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
DEVICE = "cuda"
NUM_CLASSES = 19
NUM_ROUNDS = 1600
NUM_CLIENTS_FOR_ROUND = 5

T = 5
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

# def compute_mIoU(y_true, y_pred,n_classes):
#      # ytrue, ypred is a flatten vector
#      y_pred = y_pred.cpu().flatten()
#      y_true = y_true.cpu().flatten()
#      labels = range(19)
#      current = confusion_matrix(y_true, y_pred, labels=labels)
#      # compute mean iou
#      intersection = np.diag(current)
#      ground_truth_set = current.sum(axis=1)
#      predicted_set = current.sum(axis=0)
#      union = ground_truth_set + predicted_set - intersection
#      IoU = intersection / union.astype(np.float32)
#      return np.mean(IoU)  

class Server():
    
    def __init__(self,clients,path,test_dataloader: DataLoader,pseudo_lab=False):
        self.test_dataloader=test_dataloader
        self.clients = clients
        self.main_model = BiSeNetV2(n_classes=19,output_aux=True,pretrained=True)
        if pseudo_lab:
          self.main_model.load_state_dict(path,strict=False)
        
        self.pseudo_lab=pseudo_lab
    

    def load_server_model_on_client(self,client):
      client.bisenet_model.load_state_dict(deepcopy(self.main_model.state_dict()))

    def _server_opt(self, pseudo_gradient):

        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.optimizer.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        return averaged_sol_n

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        self.writer.write(f"total grad norm: {round(total_grad, 2)}")  # 0: no gradients server side
        return total_grad

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        averaged_sol_n = self._aggregation()

        if self.optimizer is not None:  # optimizer step
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n, strict=False)
        self.model_params_dict = deepcopy(self.model.state_dict())

        self.updates = []


    def evaluate(self):
      net = self.main_model.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
      net.train(False) # Set Network to evaluation mode
      net=net.half()
      running_corrects = 0
      torch.cuda.empty_cache() 
      mIoU = 0
      count = 0
      for images, labels in self.test_dataloader:
        images = images.half().to(DEVICE)
        labels = labels.half().to(DEVICE)
        #print("images:"+str(images.size()))
        #print("labels:"+str(labels.size()))
        # Forward Pass
        outputs = net(images,test=True,use_test_resize=False)
        preds = outputs.argmax(dim=1)
        # print(outputs.size())
        mIoU += compute_mIoU(labels,preds)
        #print(mIoU)
        count += 1
        
        # iou = ops.box_iou(labels, outputs)

        # print('IOU : ', iou.numpy()[0][0])
      print("\nmIoU = ",mIoU/count)
      net.train(True)
      return mIoU/count

    def train(self):
        check_round = 0
        mIoU_for_rounds = []
        loss_train = []

        # checkpoint = torch.load('/content/drive/MyDrive/step5/FDA/DatasetA_I/checkpoints/T1/450checkpoint.pt')
        # self.main_model.load_state_dict(checkpoint['model_state_dict'])
        # self.main_model.to(DEVICE)
        # check_round = checkpoint['round']
        # mIoU_for_rounds = checkpoint['mIoU_for_rounds']
        # loss_train = checkpoint['loss_train']

        if self.pseudo_lab:
          for client in self.clients:
            client.criterion.set_teacher(deepcopy(self.main_model).half())
        for i in range(check_round,NUM_ROUNDS):

            new_weights = []
            num_samples_list = []
            tot_samples = 0
            

            for client in random.sample(self.clients,NUM_CLIENTS_FOR_ROUND):
              if self.pseudo_lab:
                if T!=0 and i % T == 0:
                  print("update teacher model...")
                  client.criterion.set_teacher(deepcopy(self.main_model).half())
                  
              self.load_server_model_on_client(client)
              num_samples, update,loss = client.train()
              #new_weights.append((num_samples, update))

              new_weights.append(update)
              tot_samples += num_samples
              num_samples_list.append(num_samples)
              

            

            print(f"[Server]: round {i}")
            
            mean_weights = OrderedDict()
            keys = list(new_weights[0].keys())

            # Loop through the keys
            for key in keys:
                # Initialize a list to store the values corresponding to the key
                values = []
                
                # Loop through the OrderedDict objects
                for n_s,d in zip(num_samples_list,new_weights):
                    # Add the value corresponding to the key to the list
                    values.append(d[key]*n_s)
                    
                # Calculate the mean value
                mean_value = sum(values) / tot_samples
                
                # Add the key and the mean value to the mean_dict
                mean_weights[key] = mean_value
            self.main_model.load_state_dict(mean_weights,strict=False)
            
            
            if i % 5 == 0:
              mIoU_for_rounds.append(self.evaluate())
              loss_train.append(loss) 
            if i % 10 == 0:
              torch.save({
              'round': i,
              'model_state_dict': self.main_model.state_dict(),
              'mIoU_for_rounds' : mIoU_for_rounds,
              'loss_train':loss_train}, '/content/drive/MyDrive/step5/FDA/DatasetA_I/checkpoints/T5/'+ str(i) +'checkpoint.pt')
        
        return self.main_model
            
        