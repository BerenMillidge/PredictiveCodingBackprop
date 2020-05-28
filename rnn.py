import tensorflow as tf
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributions as dist
import torch.optim as optim
import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime
from utils import *
from datasets import *

def rnn_accuracy(model, target_batch):
    accuracy = 0
    L, _, B = target_batch.shape
    for i in range(len(model.y_preds)): # this loop is over the seq_len 
      for b in range(B):
        if torch.argmax(target_batch[i,:,b]) ==torch.argmax(model.y_preds[i][:,b]):
          accuracy+=1
    return accuracy / (L * B)


class PC_RNN(object):
  def __init__(self, hidden_size, input_size, output_size,batch_size,vocab_size, fn, fn_deriv,inference_learning_rate, weight_learning_rate, n_inference_steps,device="cpu"):
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.inference_learning_rate = inference_learning_rate
    self.weight_learning_rate = weight_learning_rate
    self.n_inference_steps = n_inference_steps
    self.device = device
    self.clamp_val = 50
    #weights
    self.Wh = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.hidden_size])))
    self.Wx = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.input_size])))
    self.Wy = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.output_size, self.hidden_size])))
    self.h0 = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.batch_size])))

  def copy_weights_from(self, model):
    self.Wh = model.Wh.clone()
    self.Wx = model.Wx.clone()
    self.Wy = model.Wy.clone()
    self.h0 = model.h0.clone()

  def forward_sweep(self, input_seq):
    self.hs = [[] for i in range(len(input_seq)+1)]
    self.y_preds = [[] for i in range(len(input_seq))]
    self.h_preds = [[] for i in range(len(input_seq)+1)]
    self.hs[0] = self.h0
    self.h_preds[0] = self.h0.clone()
    for i,inp in enumerate(input_seq):
      self.h_preds[i+1] = self.fn(self.Wh @ self.h_preds[i] + self.Wx @ inp)
      self.hs[i+1] = self.h_preds[i+1].clone()
      self.y_preds[i] = linear(self.Wy @ self.h_preds[i+1])

  def infer(self, input_seq, target_seq,fixed_predictions=True):
    with torch.no_grad():
      #input sequence = [list of [Batch_size x Feature_Dimension]] seq len 
      self.e_ys = [[] for i in range(len(target_seq))] #ouptut prediction errors
      self.e_hs = [[] for i in range(len(input_seq))] # hidden state prediction errors
      for i, (inp, targ) in reversed(list(enumerate(zip(input_seq,target_seq)))):
        for n in range(self.n_inference_steps):
          self.e_ys[i] =  targ - self.y_preds[i]
          if fixed_predictions == False:
            self.h_preds[i+1] = self.fn(self.Wh @ self.hs[i] + self.Wx @ inp)
          self.e_hs[i] = self.hs[i+1] - self.h_preds[i+1]
          hdelta = self.e_hs[i].clone()
          hdelta -= self.Wy.T @ (self.e_ys[i] * linear_deriv(self.Wy @ self.h_preds[i+1]))
          if i < len(target_seq)-1:
            fn_deriv =  self.fn_deriv(self.Wh @ self.h_preds[i+1] + self.Wx @ input_seq[i+1])
            hdelta -= self.Wh.T @ (self.e_hs[i+1] * fn_deriv)
          self.hs[i+1] -= self.inference_learning_rate * hdelta
          if fixed_predictions == False:
            self.y_preds[i] = linear(self.Wy @ self.hs[i+1])
      return self.e_ys, self.e_hs

  def update_weights(self, input_seq,update_weights=True):
    with torch.no_grad():
      dWy = set_tensor(torch.zeros_like(self.Wy))
      dWx = set_tensor(torch.zeros_like(self.Wx))
      dWh = set_tensor(torch.zeros_like(self.Wh))
      for i in reversed(list(range(len(input_seq)))):
        fn_deriv = self.fn_deriv(self.Wh @ self.h_preds[i] + (self.Wx @ input_seq[i]))
        dWy += (self.e_ys[i] * linear_deriv(self.Wy @ self.h_preds[i+1])) @ self.h_preds[i+1].T #if self.e_ys[i] is not None else torch.zeros_like(self.Wy)
        dWx += (self.e_hs[i] * fn_deriv) @ input_seq[i].T
        dWh += (self.e_hs[i] * fn_deriv) @ self.h_preds[i].T
      if update_weights:
        self.Wy += self.weight_learning_rate * torch.clamp(dWy,-self.clamp_val,self.clamp_val)
        self.Wx += self.weight_learning_rate * torch.clamp(dWx,-self.clamp_val, self.clamp_val)
        self.Wh += self.weight_learning_rate * torch.clamp(dWh,-self.clamp_val, self.clamp_val)
      return dWy, dWx, dWh

  def decode_predictions(self, y_preds,target_list):
    chars = decode_ypreds(y_preds)
    target_chars = inverse_onehot(target_list)
    print(chars[:,0])
    print(target_chars[:,0])
    return chars, target_chars

  def save_model(self, logdir, savedir,losses=None, accs=None):
    np.save(logdir + "/Wh.npy", self.Wh.detach().cpu().numpy())
    np.save(logdir + "/Wx.npy", self.Wx.detach().cpu().numpy())
    np.save(logdir + "/Wy.npy", self.Wy.detach().cpu().numpy())
    np.save(logdir + "/h0.npy", self.h0.detach().cpu().numpy())
    if losses is not None:
        np.save(logdir+ "/losses.npy", np.array(losses))
    if accs is not None:
        np.save(logdir+"/accs.npy", np.array(accs))

    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(self, save_dir):
        Wh = np.load(save_dir+"/Wh.npy")
        self.Wh = set_tensor(torch.from_numpy(Wh))
        Wx = np.load(save_dir+"/Wx.npy")
        self.Wx = set_tensor(torch.from_numpy(Wx))
        Wy = np.load(save_dir+"/Wy.npy")
        self.Wy = set_tensor(torch.from_numpy(Wy))
        h0 = np.load(save_dir+"/h0.npy")
        self.h0 = set_tensor(torch.from_numpy(h0))
  
  def train(self,dataset,n_epochs,logdir,savedir,seq_length,old_savedir="None",save_every=1):
    with torch.no_grad():
      if old_savedir != "None":
          self.load_model(savedir)
      losses = []
      accs =[]
      for n in range(n_epochs):
        print("Epoch: ",n)
        for i,(inp, target) in enumerate(dataset):
          input_seq = set_tensor(torch.from_numpy(onehot(inp.reshape(seq_length,self.batch_size),self.vocab_size)))
          target_seq = set_tensor(torch.from_numpy(onehot(target.reshape(seq_length,self.batch_size),self.vocab_size)))
          self.forward_sweep(input_seq)
          self.infer(input_seq, target_seq)
          self.update_weights(input_seq)
          if i % save_every == 0:
            loss = np.sum(np.array([torch.sum((self.y_preds[t]-target_seq[t])**2).item() for t in range(len(input_seq))]))
            print("Loss Epoch " + str(n) + " batch " + str(i) + ": " + str(loss))
            acc = rnn_accuracy(self,target_seq)
            print("Accuracy: ", acc)
            losses.append(loss)
            accs.append(acc)
            #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char = True)),file=output_file)
            #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char=False)),file=output_file)
          if i % 200 == 0:
            print("FINISHED EPOCH: " + str(n) + " SAVING MODEL")
            self.save_model(logdir, savedir,losses,accs)
      return self.y_preds
        
class Backprop_RNN(object):
  def __init__(self, hidden_size, input_size, output_size,batch_size,vocab_size, fn, fn_deriv,learning_rate):
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.clamp_val = 50
    #weights
    self.Wh = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.hidden_size])))
    self.Wx = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.input_size])))
    self.Wy = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.output_size, self.hidden_size])))
    self.h0 = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_size, self.batch_size])))

  def copy_weights_from(self, model):
    self.Wh = model.Wh.clone()
    self.Wx = model.Wx.clone()
    self.Wy = model.Wy.clone()
    self.h0 = model.h0.clone()

  def forward_sweep(self, input_seq):
    self.hs = [[] for i in range(len(input_seq)+1)]
    self.y_preds = [[] for i in range(len(input_seq))] 
    self.hs[0] = self.h0
    for i,inp in enumerate(input_seq):
      self.hs[i+1] = self.fn(self.Wh @ self.hs[i] + self.Wx @ inp)
      self.y_preds[i] = linear(self.Wy @ self.hs[i+1])
    return self.y_preds

  def backward_sweep(self,input_seq, target_seq):
    self.dys = [[] for i in range(len(input_seq))]
    self.dhs = [[] for i in range(len(input_seq)+1)]
    for i, (inp, targ) in reversed(list(enumerate(zip(input_seq, target_seq)))):
      self.dys[i] = targ - self.y_preds[i]
      dhdh = self.Wy.T @ (self.dys[i] * linear_deriv(self.Wy @ self.hs[i+1]))
      if i < len(target_seq) -1:
        fn_deriv =  self.fn_deriv(self.Wh @ self.hs[i+1] + self.Wx @ input_seq[i+1])
        dhdh += self.Wh.T @ (self.dhs[i+1] * fn_deriv)
      self.dhs[i]= dhdh
    return self.dhs, self.dys

  def update_weights(self,input_seq,update_weights=True):
    dWy = torch.zeros_like(self.Wy)
    dWx = torch.zeros_like(self.Wx)
    dWh = torch.zeros_like(self.Wh)
    for i,inp in reversed(list(enumerate(input_seq))):
      fn_deriv = self.fn_deriv(self.Wh @ self.hs[i] + self.Wx @ input_seq[i])
      dWy += (self.dys[i] * linear_deriv(self.Wy @ self.hs[i+1])) @ self.hs[i+1].T
      dWx += (self.dhs[i] * fn_deriv) @ inp.T
      dWh += (self.dhs[i] * fn_deriv) @ self.hs[i].T
    if update_weights:
      self.Wy += self.learning_rate * torch.clamp(dWy,-self.clamp_val,self.clamp_val)
      self.Wx += self.learning_rate * torch.clamp(dWx,-self.clamp_val, self.clamp_val)
      self.Wh += self.learning_rate * torch.clamp(dWh,-self.clamp_val, self.clamp_val)
    return dWy, dWx, dWh

  def decode_predictions(self, y_preds,target_list):
    chars = decode_ypreds(y_preds)
    target_chars = inverse_onehot(target_list)
    print(chars[:,0])
    print(target_chars[:,0])
    return chars, target_chars

  def save_model(self, logdir, savedir,losses=None, accs=None):
    np.save(logdir + "/Wh.npy", self.Wh.detach().cpu().numpy())
    np.save(logdir + "/Wx.npy", self.Wx.detach().cpu().numpy())
    np.save(logdir + "/Wy.npy", self.Wy.detach().cpu().numpy())
    np.save(logdir + "/h0.npy", self.h0.detach().cpu().numpy())
    if losses is not None:
        np.save(logdir+ "/losses.npy", np.array(losses))
    if accs is not None:
        np.save(logdir+"/accs.npy", np.array(accs))

    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(self, save_dir):
        Wh = np.load(save_dir+"/Wh.npy")
        self.Wh = set_tensor(torch.from_numpy(Wh))
        Wx = np.load(save_dir+"/Wx.npy")
        self.Wx = set_tensor(torch.from_numpy(Wx))
        Wy = np.load(save_dir+"/Wy.npy")
        self.Wy = set_tensor(torch.from_numpy(Wy))
        h0 = np.load(save_dir+"/h0.npy")
        self.h0 = set_tensor(torch.from_numpy(h0))

  def train(self,dataset,n_epochs,logdir,savedir,seq_length,old_savedir="None",save_every=1):
    with torch.no_grad():
      if old_savedir != "None":
          self.load_model(savedir)
      losses = []
      accs = []
      for n in range(n_epochs):
        print("Epoch: ",n)
        for i,(inp, target) in enumerate(dataset):
          input_seq = set_tensor(torch.from_numpy(onehot(inp.reshape(seq_length,self.batch_size),self.vocab_size)))
          target_seq = set_tensor(torch.from_numpy(onehot(target.reshape(seq_length,self.batch_size),self.vocab_size)))
          self.forward_sweep(input_seq)
          self.backward_sweep(input_seq, target_seq)
          dWy,dWx,dWh = self.update_weights(input_seq)
          #print("gradients: ", torch.mean(torch.abs(dWy)), torch.mean(torch.abs(dWx)), torch.mean(torch.abs(dWh)))
          if i % save_every == 0:
            loss = np.sum(np.array([torch.sum((self.y_preds[t]-target_seq[t])**2).item() for t in range(len(input_seq))]))
            print("Loss Epoch " + str(n) + " batch " + str(i) + ": " + str(loss))
            acc = rnn_accuracy(self,target_seq)
            print("Accuracy: ", acc)
            losses.append(loss)
            accs.append(acc)
            #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char = True)),file=output_file)
            #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char=False)),file=output_file)
          if i % 200 == 0:
            print("FINISHED EPOCH: " + str(n) + " SAVING MODEL")
            self.save_model(logdir, savedir,losses,accs)
      return self.y_preds


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    print("Initialized")
        #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--seq_len",type=int,default=50)
    parser.add_argument("--hidden_size",type=int,default=1056)
    parser.add_argument("--n_inference_steps",type=int, default=200)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--weight_learning_rate",type=float,default=0.0001)
    parser.add_argument("--N_epochs",type=int, default=10000)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--network_type",type=str,default="backprop")
    parser.add_argument("--old_savedir",type=str,default="None")

    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    dataset, vocab_size,char2idx,idx2char = get_lstm_dataset(args.seq_len, args.batch_size)
    print("dataset loaded")
    dataset = [[inp.numpy(),target.numpy()] for (inp, target) in dataset]
    print("dataset numpified")

    input_size = vocab_size
    hidden_size = args.hidden_size
    output_size = vocab_size
    batch_size = args.batch_size
    inference_learning_rate = args.inference_learning_rate
    weight_learning_rate = args.weight_learning_rate
    n_inference_steps = args.n_inference_steps
    n_epochs = args.N_epochs
    save_every = args.save_every

    #define networks
    if args.network_type == "pc":
        net = PC_RNN(hidden_size, input_size,output_size,batch_size,vocab_size,tanh, tanh_deriv,inference_learning_rate,weight_learning_rate/2,n_inference_steps)
    elif args.network_type == "backprop":
        net = Backprop_RNN(hidden_size,input_size,output_size,batch_size,vocab_size,tanh, tanh_deriv,weight_learning_rate)
    else:
        raise Exception("Unknown network type entered")

    #train!
    net.train(dataset, int(n_epochs),args.logdir, args.savedir,args.seq_len,old_savedir=args.old_savedir,save_every=args.save_every)




