import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2

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


## IOU in pure numpy
def numpy_iou(y_true, y_pred, n_class=2):
  def iou(y_true, y_pred, n_class):
      # IOU = TP/(TP+FN+FP)
      IOU = []
      for c in range(n_class):
          TP = np.sum((y_true == c) & (y_pred == c))
          FP = np.sum((y_true != c) & (y_pred == c))
          FN = np.sum((y_true == c) & (y_pred != c))

          n = TP
          d = float(TP + FP + FN + 1e-12)

          iou = np.divide(n, d)
          IOU.append(iou)

      return np.mean(IOU)

  batch = y_true.shape[0]
  y_true = np.reshape(y_true, (batch, -1))
  y_pred = np.reshape(y_pred, (batch, -1))

  score = []
  for idx in range(batch):
      iou_value = iou(y_true[idx], y_pred[idx], n_class)
      score.append(iou_value)
  return np.mean(score)

def numpy_mean_iou(y_true, y_pred):
  prec = []
  score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
  prec.append(score)
  return K.mean(K.stack(prec), axis=0)





