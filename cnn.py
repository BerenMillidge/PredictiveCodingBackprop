import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime
from datasets import *
from utils import * 
from layers import *


class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate,device='cpu',numerical_check=False):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.device = device
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.numerical_check = numerical_check
    if self.numerical_check:
      print("Numerical Check Activated!")
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        if self.numerical_check:
            true_weight_grad = l.get_true_weight_grad().clone()
        dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
        true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
        diff = torch.sum((dW -true_dW)**2).item()
        weight_diffs.append(diff)
        if print_weight_grads:
          print("weight grads : ", i)
          print("dW: ", dW*2)
          print("true diffs: ", true_dW * 2)
          if self.numerical_check:
            print("true weights ", true_weight_grad)
    return weight_diffs


  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp,label,n_inference_steps=None):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i,l in enumerate(self.layers):
        #initialize mus with forward predictions
        self.mus[i+1] = l.forward(self.mus[i])
        self.outs[i+1] = self.mus[i+1].clone()
      self.mus[-1] = label.clone() #setup final label
      self.prediction_errors[-1] = self.mus[-1] - self.outs[-1] #setup final prediction errors
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
      #reversed inference 
        for j in reversed(range(len(self.layers))):
          if j != 0: 
            self.prediction_errors[j] = self.mus[j] - self.outs[j]
            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
            dx_l = self.prediction_errors[j] - self.predictions[j]
            self.mus[j] -= self.inference_learning_rate * (2*dx_l)
      #update weights
      weight_diffs = self.update_weights()
      #get loss:
      L = torch.sum(self.prediction_errors[-1]**2).item()
      #get accuracy
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def test_accuracy(self,testset):
    accs = [] 
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self,dataset,testset,n_epochs,n_inference_steps,logdir,savedir, old_savedir,save_every=1,print_every=10):
    if old_savedir != "None":
      self.load_model(old_savedir)
    losses = []
    accs = []
    weight_diffs_list = []
    test_accs = []
    for epoch in range(n_epochs):
      losslist = []
      print("Epoch: ", epoch)
      for i,(inp, label) in enumerate(dataset):
        L, acc,weight_diffs = self.infer(inp.to(DEVICE),onehot(label).to(DEVICE))
        losslist.append(L)
      mean_acc, acclist = self.test_accuracy(dataset)
      accs.append(mean_acc)
      mean_loss = np.mean(np.array(losslist))
      losses.append(mean_loss)
      mean_test_acc, _ = self.test_accuracy(testset)
      test_accs.append(mean_test_acc)
      weight_diffs_list.append(weight_diffs)
      print("TEST ACCURACY: ", mean_test_acc)
      print("SAVING MODEL")
      self.save_model(logdir,savedir,losses,accs,weight_diffs_list,test_accs)

  def save_model(self,savedir,logdir,losses,accs,weight_diffs_list,test_accs):
      for i,l in enumerate(self.layers):
          l.save_layer(logdir,i)
      np.save(logdir +"/losses.npy",np.array(losses))
      np.save(logdir+"/accs.npy",np.array(accs))
      np.save(logdir+"/weight_diffs.npy",np.array(weight_diffs_list))
      np.save(logdir+"/test_accs.npy",np.array(test_accs))
      subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
      print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
      now = datetime.now()
      current_time = str(now.strftime("%H:%M:%S"))
      subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(self,old_savedir):
      for (i,l) in enumerate(self.layers):
          l.load_layer(old_savedir,i)


class Backprop_CNN(object):
  def __init__(self, layers):
    self.layers = layers 
    self.xs = [[] for i in range(len(self.layers)+1)]
    self.e_ys = [[] for i in range(len(self.layers)+1)]
    for l in self.layers:
      l.set_weight_parameters()

  def forward(self, inp):
    self.xs[0] = inp
    for i,l in enumerate(self.layers):
      self.xs[i+1] = l.forward(self.xs[i])
    return self.xs[-1]

  def backward(self,e_y):
    self.e_ys[-1] = e_y
    for (i,l) in reversed(list(enumerate(self.layers))):
      self.e_ys[i] = l.backward(self.e_ys[i+1])
    return self.e_ys[0]

  def update_weights(self,print_weight_grads=False,update_weight=False,sign_reverse=False):
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.e_ys[i+1],update_weights=update_weight,sign_reverse=sign_reverse)
      if print_weight_grads:
        print("weight grads : ", i)
        print("dW: ", dW*2)
        print("weight grad: ",l.get_true_weight_grad())

  def save_model(self,savedir,logdir,losses,accs,test_accs):
      for i,l in enumerate(self.layers):
          l.save_layer(logdir,i)
      np.save(logdir +"/losses.npy",np.array(losses))
      np.save(logdir+"/accs.npy",np.array(accs))
      np.save(logdir+"/test_accs.npy",np.array(test_accs))
      subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
      print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
      now = datetime.now()
      current_time = str(now.strftime("%H:%M:%S"))
      subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(old_savedir):
      for (i,l) in enumerate(self.layers):
          l.load_layer(old_savedir,i)

  def test_accuracy(self,testset):
    accs = [] 
    for i,(inp, label) in enumerate(testset):
        pred_y = self.forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self, dataset,testset,n_epochs,n_inference_steps,savedir,logdir,old_savedir="",print_every=100,save_every=1):
    if old_savedir != "None":
        self.load_model(old_savedir)
    with torch.no_grad():
      accs = []
      losses = []
      test_accs =[]
      for n in range(n_epochs):
        print("Epoch: ",n)
        losslist = []
        for (i,(inp,label)) in enumerate(dataset):
          out = self.forward(inp.to(DEVICE))
          label = onehot(label).to(DEVICE)
          e_y = out - label
          self.backward(e_y)
          self.update_weights(update_weight=True,sign_reverse=True)
          loss = torch.sum(e_y**2).item()
          losslist.append(loss)
        mean_acc, acclist = self.test_accuracy(dataset)
        accs.append(mean_acc)
        mean_loss = np.mean(np.array(losslist))
        losses.append(mean_loss)
        mean_test_acc, _ = self.test_accuracy(testset)
        test_accs.append(mean_test_acc)
        print("ACCURACY: ", mean_acc)
        print("TEST ACCURACY: ", mean_test_acc)
        print("SAVING MODEL")
        self.save_model(logdir,savedir,losses,accs,test_accs)



if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--learning_rate",type=float,default=0.0005)
    parser.add_argument("--N_epochs",type=int, default=100)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--print_every",type=int,default=10)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--network_type",type=str,default="pc")
    parser.add_argument("--dataset",type=str,default="cifar")

    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    dataset,testset = get_cnn_dataset(args.dataset,args.batch_size)

    if args.dataset in ["cifar", "mnist","svhn"]:
        output_size = 10
    if args.dataset == "cifar100":
        output_size=100

    def onehot(x):
        z = torch.zeros([len(x),output_size])
        for i in range(len(x)):
            z[i,x[i]] = 1
        return z.float().to(DEVICE)
    #l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l2 = MaxPool(2,device=DEVICE)
    #l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,args.learning_rate,device=DEVICE)
    #l5 = FCLayer(120,84,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l6 = FCLayer(84,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #layers =[l1,l2,l3,l4,l5,l6]
    l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),200,relu,relu_deriv,args.learning_rate,device=DEVICE)
    l5 = FCLayer(200,150,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l6 = FCLayer(150,output_size,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    layers =[l1,l2,l3,l4,l5,l6]
    #l1 = ConvLayer(32,3,20,64,4,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
    #l2 = ConvLayer(29,20,50,64,5,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
    #l3 = ConvLayer(25,50,50,64,5,args.learning_rate,tanh,tanh_deriv,stride=2,padding=1,device=DEVICE)
    #l4 = ConvLayer(12,50,5,64,3,args.learning_rate,tanh,tanh_deriv,stride=1,device=DEVICE)
    #l5 = ProjectionLayer((64,5,10,10),200,sigmoid,sigmoid_deriv,args.learning_rate,device=DEVICE)
    #l6 = FCLayer(200,100,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #l7 = FCLayer(100,50,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #l8 = FCLayer(50,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #layers =[l1,l2,l3,l4,l5,l6,l7,l8]
    if args.network_type == "pc":
        net = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,device=DEVICE)
    elif args.network_type == "backprop":
        net = Backprop_CNN(layers)
    else:
        raise Exception("Network type not recognised: must be one of 'backprop', 'pc'")
    net.train(dataset[0:-2],testset[0:-2],args.N_epochs,args.n_inference_steps,args.savedir,args.logdir,args.old_savedir,args.save_every,args.print_every)


