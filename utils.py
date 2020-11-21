import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributions as dist
from copy import deepcopy
import math
import matplotlib.pyplot as plt

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### General Utils ###
def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

def set_tensor(xs):
    return xs.float().to(DEVICE)

def edge_zero_pad(img,d):
  N,C, h,w = img.shape
  x = torch.zeros((N,C,h+(d*2),w+(d*2))).to(DEVICE)
  x[:,:,d:h+d,d:w+d] = img
  return x

def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B

def sequence_accuracy(model, target_batch):
    accuracy = 0
    L = len(target_batch)
    _,B = target_batch[0].shape
    s = ""
    for i in range(len(target_batch)): # this loop is over the seq_len
      s += str(torch.argmax(model.mu_y[i][:,0]).item()) + " " + str(torch.argmax(target_batch[i][:,0]).item()) + "  "
      for b in range(B):
        #print("target idx: ", torch.argmax(target_batch[i][:,b]).item())
        #print("pred idx: ", torch.argmax(model.mu_y[i][:,b]).item())
        if torch.argmax(target_batch[i][:,b]) ==torch.argmax(model.mu_y[i][:,b]):
          accuracy+=1
    print("accs: ", s)
    return accuracy / (L * B)

def custom_onehot(idx, shape):
  ret = set_tensor(torch.zeros(shape))
  ret[idx] =1
  return ret

def onehot(arr, vocab_size):
  L, B = arr.shape
  ret = np.zeros([L,vocab_size,B])
  for l in range(L):
    for b in range(B):
      ret[l,int(arr[l,b]),b] = 1
  return ret

def inverse_list_onehot(arr):
  L = len(arr)
  V,B = arr[0].shape
  ret = np.zeros([L,B])
  for l in range(L):
    for b in range(B):
      for v in range(V):
        if arr[l][v,b] == 1:
          ret[l,b] = v
  return ret

def decode_ypreds(ypreds):
  L = len(ypreds)
  V,B = ypreds[0].shape
  ret = np.zeros([L,B])
  for l in range(L):
    for b in range(B):
      v = torch.argmax(ypreds[l][:,b])
      ret[l,b] =v
  return ret


def inverse_onehot(arr):
  if type(arr) == list:
    return inverse_list_onehot(arr)
  else:
    L,V,B = arr.shape
    ret = np.zeros([L,B])
    for l in range(L):
      for b in range(B):
        for v in range(V):
          if arr[l,v,b] == 1:
            ret[l,b] = v
    return ret

### Activation functions ###
def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones((1,)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel

def softmax(xs):
  return torch.nn.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))


### loss functions
def mse_loss(out, label):
      return torch.sum((out-label)**2)

def mse_deriv(out,label):
      return 2 * (out - label)

ce_loss = nn.CrossEntropyLoss()

def cross_entropy_loss(out,label):
      return ce_loss(out,label)

def my_cross_entropy(out,label):
      return -torch.sum(label * torch.log(out + 1e-6))

def cross_entropy_deriv(out,label):
      return out - label

def parse_loss_function(loss_arg):
      if loss_arg == "mse":
            return mse_loss, mse_deriv
      elif loss_arg == "crossentropy":
            return my_cross_entropy, cross_entropy_deriv
      else:
            raise ValueError("loss argument not expected. Can be one of 'mse' and 'crossentropy'. You inputted " + str(loss_arg))


### Initialization Functions ###
def gaussian_init(W,mean=0.0, std=0.05):
  return W.normal_(mean=0.0,std=0.05)

def zeros_init(W):
  return torch.zeros_like(W)

def kaiming_init(W, a=math.sqrt(5),*kwargs):
  return init.kaiming_uniform_(W, a)

def glorot_init(W):
  return init.xavier_normal_(W)

def kaiming_bias_init(b,*kwargs):
  fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
  bound = 1 / math.sqrt(fan_in)
  return init.uniform_(b, -bound, bound)

#the initialization pytorch uses for lstm
def std_uniform_init(W,hidden_size):
  stdv = 1.0 / math.sqrt(hidden_size)
  return init.uniform_(W, -stdv, stdv)
