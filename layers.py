
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from utils import *

class ConvLayer(object):
  def __init__(self,input_size,num_channels,num_filters,batch_size,kernel_size,learning_rate,f,df,padding=0,stride=1,device="cpu"):
    self.input_size = input_size
    self.num_channels = num_channels
    self.num_filters = num_filters
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) +1
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.device = device
    self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
    self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
    self.fold = nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)

  def forward(self,inp):
    self.X_col = self.unfold(inp.clone())
    self.flat_weights = self.kernel.reshape(self.num_filters,-1)
    out = self.flat_weights @ self.X_col
    self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
    return self.f(self.activations)

  def update_weights(self,e,update_weights=False):
    fn_deriv = self.df(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dW = self.dout @ self.X_col.permute(0,2,1)
    dW = torch.sum(dW,dim=0)
    dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size,self.kernel_size))
    if update_weights:
      self.kernel += self.learning_rate * torch.clamp(dW * 2,-50,50)
    return dW

  def backward(self,e):
    fn_deriv = self.df(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dX_col = self.flat_weights.T @ self.dout
    dX = self.fold(dX_col)
    return torch.clamp(dX,-50,50)

  def get_true_weight_grad(self):
    return self.kernel.grad

  def set_weight_parameters(self):
    self.kernel = nn.Parameter(self.kernel)

  def save_layer(self,logdir,i):
      np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.kernel.detach().cpu().numpy())
  
  def load_layer(self,logdir,i):
    kernel = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.kernel = set_tensor(torch.from_numpy(kernel))

class MaxPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)

  def forward(self,x):
    out, self.idxs = F.max_pool2d(x, self.kernel_size,return_indices=True)
    #print(out.shape)
    return out
  
  def backward(self, y):
    return F.max_unpool2d(y,self.idxs, self.kernel_size)

  def update_weights(self,e,update_weights=False):
    return 0

  def get_true_weight_grad(self):
    return None

  def set_weight_parameters(self):
    pass

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass

class AvgPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)
  
  def forward(self, x):
    self.B_in,self.C_in,self.H_in,self.W_in = x.shape
    return F.avg_pool2d(x,self.kernel_size)

  def backward(self, y):
    N,C,H,W = y.shape
    print("in backward: ", y.shape)
    return F.interpolate(y,scale_factor=(1,1,self.kernel_size,self.kernel_size))

  def update_weights(self,x):
    return 0

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass



class ProjectionLayer(object):
  def __init__(self,input_size, output_size,f,df,learning_rate,device='cpu'):
    self.input_size = input_size
    self.B, self.C, self.H, self.W = self.input_size
    self.output_size =output_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.device = device
    self.Hid = self.C * self.H * self.W
    self.weights = torch.empty((self.Hid, self.output_size)).normal_(mean=0.0, std=0.05).to(self.device)

  def forward(self, x):
    self.inp = x.detach().clone()
    out = x.reshape((len(x), -1))
    self.activations = torch.matmul(out,self.weights)
    return self.f(self.activations)

  def backward(self, e):
    fn_deriv = self.df(self.activations)
    out = torch.matmul(e * fn_deriv, self.weights.T)
    out = out.reshape((len(e), self.C, self.H, self.W))
    return torch.clamp(out,-50,50)

  def update_weights(self, e,update_weights=False):
    out = self.inp.reshape((len(self.inp), -1))
    fn_deriv = self.df(self.activations)
    dw = torch.matmul(out.T, e * fn_deriv)
    if update_weights:
      self.weights += self.learning_rate * torch.clamp((dw * 2),-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))

class FCLayer(object):
  def __init__(self, input_size,output_size,batch_size, learning_rate,f,df,device="cpu"):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f 
    self.df = df
    self.device = device
    self.weights = torch.empty([self.input_size,self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self,x):
    #self.inp = x.detach()
    self.inp = x.clone()
    self.activations = torch.matmul(self.inp, self.weights)
    return self.f(self.activations)

  def backward(self,e):
    self.fn_deriv = self.df(self.activations)
    out = torch.matmul(e * self.fn_deriv, self.weights.T)
    return torch.clamp(out,-50,50)

  def update_weights(self,e,update_weights=False):
    self.fn_deriv = self.df(self.activations)
    #print("fnderiv: ", self.fn_deriv.shape)
    #print("inp: ", self.inp.T.shape)
    #print("inputs; ", e.shape)
    dw = torch.matmul(self.inp.T, e * self.fn_deriv)
    if update_weights:
      self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))