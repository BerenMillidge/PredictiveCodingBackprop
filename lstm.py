
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


class PC_LSTM(object):
  def __init__(self,input_dim, hidden_dim,output_dim,vocab_size, batch_size, inference_learning_rate,weight_learning_rate, n_inference_steps_train,weight_init=gaussian_init, bias_init=zeros_init):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim 
    self.output_dim = output_dim
    self.vocab_size = vocab_size
    self.batch_size = batch_size 
    self.inference_learning_rate = inference_learning_rate 
    self.weight_learning_rate = weight_learning_rate
    self.clamp_val = 50
    self.weight_init = weight_init
    self.bias_init = bias_init
    self.n_inference_steps_train = n_inference_steps_train 
    self.z_dim = self.input_dim + self.hidden_dim
    #initialize weights
    self.Wf = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wi = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wc = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wo = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wy = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.output_dim, self.hidden_dim])))
    #initialize biases 
    self.bf = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bi = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bc = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bo = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.by = set_tensor(torch.zeros((self.output_dim,1)))
    self.embed = nn.Embedding(self.vocab_size, self.input_dim).to(DEVICE)
    self.zero_gradients()

  def zero_gradients(self):
    #reset all gradients accumulated once the sequence is over
    #zero weights
    self.dWf = set_tensor(torch.zeros_like(self.Wf))
    self.dWi = set_tensor(torch.zeros_like(self.Wi))
    self.dWc = set_tensor(torch.zeros_like(self.Wc))
    self.dWo = set_tensor(torch.zeros_like(self.Wo))
    self.dWy = set_tensor(torch.zeros_like(self.Wy))
    #zero biases
    self.dbf = set_tensor(torch.zeros_like(self.bf))
    self.dbi = set_tensor(torch.zeros_like(self.bi))
    self.dbc = set_tensor(torch.zeros_like(self.bc))
    self.dbo = set_tensor(torch.zeros_like(self.bo))
    self.dby = set_tensor(torch.zeros_like(self.by))

  def copy_params_from(self, model):
    #copy weights
    self.dWf = set_tensor(model.dWf.clone())
    self.dWi = set_tensor(model.dWi.clone())
    self.dWc = set_tensor(model.dWc.clone())
    self.dWo = set_tensor(model.dWo.clone())
    self.dWy = set_tensor(model.dWy.clone())
    #copy biases
    self.dbf = set_tensor(model.dbf.clone())
    self.dbi = set_tensor(model.dbi.clone())
    self.dbc = set_tensor(model.dbc.clone())
    self.dbo = set_tensor(model.dbo.clone())
    self.dby = set_tensor(model.dby.clone())

  def cell_forward(self, inp, hprev, cellprev,t):
    #forward pass for a single timestep of the LSTM
    #inp = input at that timestep [Features x Batch]
    #hprev = previous hidden states [Hidden_size x Batch]
    #cellprev = previous cell state [Hidden_size x Batch]
    #t = position in sequence [int]
    #implements the forward pass of the LSTM. Saves the predictions for later inference
    embed = self.embed(inp).permute(1,0)
    z = torch.cat((embed, hprev),axis=0)
    #forget gate
    self.mu_f_activations[t] = self.Wf @ z + self.bf
    self.mu_f[t] = sigmoid(self.mu_f_activations[t])
    #input gate
    self.mu_i_activations[t] = self.Wi @ z + self.bi
    self.mu_i[t] = sigmoid(self.mu_i_activations[t])
    #control gate
    self.mu_c_activations[t] = self.Wc @ z + self.bc
    self.mu_c[t] = tanh(self.mu_c_activations[t])
    #output gate
    self.mu_o_activations[t] = self.Wo @ z + self.bo
    self.mu_o[t] = sigmoid(self.mu_o_activations[t])

    self.mu_cell[t] = torch.mul(self.mu_f[t], cellprev) + torch.mul(self.mu_i[t], self.mu_c[t])
    self.mu_h[t] = torch.mul(self.mu_o[t], tanh(self.mu_cell[t]))
    self.mu_y[t] = self.Wy @ self.mu_h[t] + self.by
    return self.mu_y[t],self.mu_h[t], self.mu_cell[t]


  def cell_backward(self,inp,true_labels,cellprev,hprev,dc_back, dh_back,t):
    # runs backward inference for a single LSTM timestep. Returns the backwards gradients
    #takes as input: inputs [Features x Batch] , labels [1 x Batch], cellprev, hprev,dc_batck, dh_back: [Hidden_size x Batch], t = timestep in seqlen [int]
    #initialize with the forward predictions
    embed = self.embed(inp).permute(1,0)
    z = torch.cat((embed, hprev),axis=0)
    h = self.mu_h[t].clone()
    o = self.mu_o[t].clone()
    c = self.mu_c[t].clone()
    cell = self.mu_cell[t].clone()
    i = self.mu_i[t].clone()
    f = self.mu_f[t].clone()
    e_y = true_labels - self.mu_y[t]
  
    for n in range(self.n_inference_steps_train):
      #compute prediction errors
      e_h = h - self.mu_h[t]
      e_o = o - self.mu_o[t]
      e_c = c - self.mu_c[t]
      e_cell = cell - self.mu_cell[t]
      e_i = i - self.mu_i[t]
      e_f = f - self.mu_f[t]
      #compute gradients
      dh = e_h -(self.Wy.T @ e_y) - dh_back  #any activation function on the output?
      do = e_o - torch.mul(sigmoid_deriv(self.mu_o_activations[t]),torch.mul(tanh(self.mu_cell[t]), e_h))
      dcell = e_cell - (self.mu_o[t] * e_h * tanh_deriv(self.mu_cell[t])) - dc_back
      dc = e_c - torch.mul(tanh_deriv(self.mu_c_activations[t]),torch.mul(e_cell,self.mu_i[t]))
      di = e_i - torch.mul(sigmoid_deriv(self.mu_i_activations[t]),torch.mul(e_cell, self.mu_c[t]))
      df = e_f - torch.mul(sigmoid_deriv(self.mu_f_activations[t]), torch.mul(cellprev, e_cell))
      #gradient updates
      h -= self.inference_learning_rate * dh
      o -= self.inference_learning_rate * do
      cell -= self.inference_learning_rate * dcell
      c -= self.inference_learning_rate * dc
      i -= self.inference_learning_rate * di
      f -= self.inference_learning_rate * df

    #accumulate weights
    self.dWf -= e_f @ z.T
    self.dWi -= e_i @ z.T
    self.dWc -= e_c @ z.T
    self.dWo -= e_o @ z.T
    self.dWy -= e_y @ self.mu_h[t].T

    #accumulate biases
    self.dbf -= torch.sum(e_f, dim=1,keepdim=True)
    self.dbi -= torch.sum(e_i, dim=1,keepdim=True)
    self.dbc -= torch.sum(e_c, dim=1,keepdim=True)
    self.dbo -= torch.sum(e_o, dim=1,keepdim=True)
    self.dby -= torch.sum(e_y, dim=1,keepdim=True)

    #compute backwards derivatives
    dc_back = self.mu_f[t] * e_cell 
    dh_back = (self.Wf.T @ e_f) +(self.Wi.T @ e_i) + (self.Wo.T @ e_o) + (self.Wc.T @ e_c)
    return dc_back, dh_back

  def update_parameters(self):
    #update weights
    self.Wi -= 1 * self.weight_learning_rate * torch.clamp(self.dWi,min=-self.clamp_val,max=self.clamp_val)
    self.Wc -= 1 * self.weight_learning_rate * torch.clamp(self.dWc,min=-self.clamp_val,max=self.clamp_val)
    self.Wo -= 1 * self.weight_learning_rate * torch.clamp(self.dWo,min=-self.clamp_val,max=self.clamp_val)
    self.Wy -= 1 * self.weight_learning_rate * torch.clamp(self.dWy,min=-self.clamp_val,max=self.clamp_val)
    #update biases
    self.bf -= self.weight_learning_rate * torch.clamp(self.dbf,min=-self.clamp_val,max=self.clamp_val)
    self.bi -= self.weight_learning_rate * torch.clamp(self.dbi,min=-self.clamp_val,max=self.clamp_val)
    self.bc -= self.weight_learning_rate * torch.clamp(self.dbc,min=-self.clamp_val,max=self.clamp_val)
    self.bo -= self.weight_learning_rate * torch.clamp(self.dbo,min=-self.clamp_val,max=self.clamp_val)
    self.by -= self.weight_learning_rate * torch.clamp(self.dby,min=-self.clamp_val,max=self.clamp_val)
    #zero gradients
    self.zero_gradients()

  def initialize_caches(self,T):
    #reset the caches for the new sequence
    self.mu_f_activations = [[] for i in range(T)]
    self.mu_f = [[] for i in range(T)]
    self.mu_i_activations = [[] for i in range(T)]
    self.mu_i = [[] for i in range(T)]
    self.mu_c_activations = [[] for i in range(T)]
    self.mu_c = [[] for i in range(T)]
    self.mu_o_activations = [[] for i in range(T)]
    self.mu_o = [[] for i in range(T)]
    self.mu_cell = [[] for i in range(T)]
    self.mu_h = [[] for i in range(T)]
    self.mu_y = [[] for i in range(T)]
    self.hprev = [[] for i in range(T+1)]
    self.cellprev = [[] for i in range(T+1)]

  def forward(self,input_seq,init_h=None,init_cell = None):
    #loop over the sequence to do the first forward pass
    T = len(input_seq)
    self.initialize_caches(T)
    #initialize starting hprev and cellprev
    #I found that random initializations each time added too much noise for gradients to learn successfully
    #self.hprev[0] = init_h if init_h is not None else set_tensor(torch.empty([self.hidden_dim, self.batch_size]).normal_(mean=0,std=0.1))
    #self.cellprev[0] = init_cell if init_cell is not None else set_tensor(torch.empty([self.hidden_dim,self.batch_size]).normal_(mean=0, std=0.1))
    self.hprev[0] = init_h if init_h is not None else set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
    self.cellprev[0] = init_cell if init_cell is not None else set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
    #roll forwards
    for (t, inp) in enumerate(input_seq):
      #roll forwards across the sequence
      out, self.hprev[t+1], self.cellprev[t+1] = self.cell_forward(inp, self.hprev[t], self.cellprev[t], t)
    return self.mu_y

  def backward(self, input_seq, target_seq):
    #this function loops over each element of the sequence backwards and performs backwards inference for each LSTM cell
    with torch.no_grad():
      T = len(input_seq)
      assert T == len(target_seq), "Input and target sequence must be same length"
      #initialize first backwards gradients to 0
      self.dc_back = set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
      self.dh_back = set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
      #begin the backwards loop
      for (t, (inp, targ)) in reversed(list(enumerate(zip(input_seq,target_seq)))):
        self.dc_back, dh_back = self.cell_backward(inp, targ,self.cellprev[t],self.hprev[t],self.dc_back,self.dh_back,t)
        #only get the hidden and not input part of this
        self.dh_back = dh_back[self.input_dim:,:]
        #not sure if I want to accumulate all these gradients in a loop. I don't think I have to
      return self.dc_back, self.dh_back,self.dWf, self.dWi, self.dWc,self.dWo,self.dWy

  def sample_sentence(self,input_char, n_steps,sample_char=False,init_h=None,init_cell=None,temp=20):
    input_seq = [set_tensor(torch.zeros_like(input_char)) for i in range(n_steps)]
    input_seq[0] = torch.tensor(input_char).reshape(1,)
    hprev = init_h if init_h is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    cellprev = init_cell if init_cell is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    output_str = ""
    #setup the network
    self.initialize_caches(n_steps)
    for n in range(1,n_steps):
      pred_y,hprev,cellprev = self.cell_forward(input_seq[n-1],hprev,cellprev,n)
      if sample_char:
       probs=F.softmax(pred_y.squeeze(1) * temp)
       cat = dist.Categorical(probs=probs)
       char_idx = cat.sample()
      else:
        #get the maximum char
        char_idx = torch.argmax(pred_y.squeeze(1))
        
      input_seq[n] = char_idx.reshape(1,)
      char = idx2char[char_idx]
      output_str+=char
    return output_str


  def sample_sentence(self,input_char, n_steps,sample_char=False,init_h=None,init_cell=None,temp=20):
    input_seq = [set_tensor(torch.zeros_like(input_char)) for i in range(n_steps)]
    input_seq[0] = torch.tensor(input_char).reshape(1,)
    hprev = init_h if init_h is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    cellprev = init_cell if init_cell is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    output_str = ""
    #setup the network
    self.initialize_caches(n_steps)
    for n in range(1,n_steps):
      pred_y,hprev,cellprev = self.cell_forward(input_seq[n-1],hprev,cellprev,n)
      if sample_char:
       probs=F.softmax(pred_y.squeeze(1) * temp)
       cat = dist.Categorical(probs=probs)
       char_idx = cat.sample()
      else:
        #get the maximum char
        char_idx = torch.argmax(pred_y.squeeze(1))
        
      input_seq[n] = char_idx.reshape(1,)
      chat_idx = char_idx.cpu().numpy()
      char = idx2char[chat_idx]
      output_str+=char
    return output_str

  def save_model(self, logdir, savedir,losses=None, accs=None):
    np.save(logdir + "/Wf.npy", self.Wf.detach().cpu().numpy())
    np.save(logdir + "/Wi.npy", self.Wi.detach().cpu().numpy())
    np.save(logdir + "/Wc.npy", self.Wc.detach().cpu().numpy())
    np.save(logdir + "/Wo.npy", self.Wo.detach().cpu().numpy())
    np.save(logdir + "/Wy.npy", self.Wy.detach().cpu().numpy())

    np.save(logdir + "/bf.npy", self.bf.detach().cpu().numpy())
    np.save(logdir + "/bi.npy", self.bi.detach().cpu().numpy())
    np.save(logdir + "/bc.npy", self.bc.detach().cpu().numpy())
    np.save(logdir + "/bo.npy", self.bo.detach().cpu().numpy())
    np.save(logdir + "/by.npy", self.by.detach().cpu().numpy())

    #np.save(logdir + "/init_h.npy", self.init_h.detach().cpu().numpy())
    #np.save(logdir + "/init_cell.npy", self.init_cell.detach().cpu().numpy())
    #save all the embedding parameters
    embed_params = list(self.embed.parameters())
    for (i,p) in enumerate(embed_params):
        np.save(logdir + "/embed_"+str(i)+".npy",p.detach().cpu().numpy())
    #SAVE the results to the edinburgh computer from scratch space to main space
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
        Wf = np.load(save_dir+"/Wf.npy")
        self.Wf = set_tensor(torch.from_numpy(Wf))
        Wi = np.load(save_dir+"/Wi.npy")
        self.Wi = set_tensor(torch.from_numpy(Wi))
        Wc = np.load(save_dir+"/Wc.npy")
        self.Wc = set_tensor(torch.from_numpy(Wc))
        Wo = np.load(save_dir+"/Wo.npy")
        self.Wo = set_tensor(torch.from_numpy(Wo))
        Wy = np.load(save_dir+"/Wy.npy")
        self.Wy = set_tensor(torch.from_numpy(Wy))

        bf = np.load(save_dir+"/bf.npy")
        self.bf = set_tensor(torch.from_numpy(bf))
        bi = np.load(save_dir+"/bi.npy")
        self.bi = set_tensor(torch.from_numpy(bi))
        bc = np.load(save_dir+"/bc.npy")
        self.bc = set_tensor(torch.from_numpy(bc))
        bo = np.load(save_dir+"/bo.npy")
        self.bo = set_tensor(torch.from_numpy(bo))
        by = np.load(save_dir+"/by.npy")
        self.by = set_tensor(torch.from_numpy(by))

        #init_h = np.load(save_dir + "/init_h.npy")
        #self.init_h = set_tensor(torch.from_numpy(init_h))
        #init_cell = np.load(save_dir + "/init_cell.npy")
        #self.init_cell = set_tensor(torch.from_numpy(init_cell))

        embed = np.load(save_dir +"/embed_0.npy")
        self.embed.weight = nn.Parameter(set_tensor(torch.from_numpy(embed)))

  def train(self,dataset,n_epochs,logdir,savedir,old_savedir="None",init_embed_path="None",save_every=20):
    #load initial embedding from backprop version
    if old_savedir == "None" and init_embed_path != "None":
        embed = np.load(init_embed_path)
        self.embed.weight = nn.Parameter(set_tensor(torch.from_numpy(embed)))
    if old_savedir != "None":
        self.load_model(old_savedir)
    with torch.no_grad():
      losses = []
      accs = []
      output_file = open(logdir +"/output.txt", "w")
      for n in range(n_epochs):
        print("Epoch: ", n)
        for (i, (input_seq, target_seq)) in enumerate(dataset):
          input_seq = list(torch.tensor(torch.from_numpy(input_seq.numpy()),dtype=torch.long).permute(1,0).to(DEVICE))
          target_seq = list(set_tensor(torch.from_numpy(onehot(target_seq, vocab_size)).float().permute(2,1,0)))
          pred_ys = self.forward(input_seq)
          self.backward(input_seq, target_seq)
          self.update_parameters()
          if i % save_every == 0:
            loss = np.sum(np.array([torch.sum((self.mu_y[t]-target_seq[t])**2).item() for t in range(len(input_seq))]))
            print("Loss Epoch " + str(n) + " batch " + str(i) + ": " + str(loss))
            acc = sequence_accuracy(self,target_seq)
            print("Accuracy: ", acc)
            losses.append(loss)
            accs.append(acc)
            print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char = True)),file=output_file)
            print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char=False)),file=output_file)
          if i % 200 == 0:
            print("FINISHED EPOCH: " + str(n) + " SAVING MODEL")
            self.save_model(logdir, savedir,losses,accs)
            


class Backprop_LSTM(object):
  def __init__(self,input_dim, hidden_dim,output_dim,vocab_size, batch_size, learning_rate,weight_init=gaussian_init, bias_init=zeros_init):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim 
    self.output_dim = output_dim
    self.vocab_size = vocab_size
    self.batch_size = batch_size 
    self.clamp_val=100
    self.learning_rate = learning_rate 
    self.weight_init = weight_init
    self.bias_init = bias_init
    self.z_dim = self.input_dim + self.hidden_dim
    #initialize weights
    #self.Wf = set_tensor(self.std_uniform_init(torch.empty([self.hidden_dim, self.z_dim]))) * 10
    #self.Wi = set_tensor(self.std_uniform_init(torch.empty([self.hidden_dim, self.z_dim]))) * 10
    #self.Wc = set_tensor(self.std_uniform_init(torch.empty([self.hidden_dim, self.z_dim]))) * 10
    #self.Wo = set_tensor(self.std_uniform_init(torch.empty([self.hidden_dim, self.z_dim]))) * 10
    #self.Wy = set_tensor(self.std_uniform_init(torch.empty([self.output_dim, self.hidden_dim]))) * 10
    #self.embed = set_tensor(self.std_uniform_init(torch.empty([self.z_dim, self.z_dim]))) 
    #
    self.Wf = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wi = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wc = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wo = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.hidden_dim, self.z_dim])))
    self.Wy = set_tensor(torch.from_numpy(np.random.normal(0,0.05,[self.output_dim, self.hidden_dim])))
    #initialize biases 
    #self.bf = set_tensor(self.std_uniform_init(torch.empty(([self.hidden_dim,1]))))
    #self.bi = set_tensor(self.std_uniform_init(torch.empty(([self.hidden_dim,1]))))
    #self.bc = set_tensor(self.std_uniform_init(torch.empty(([self.hidden_dim,1]))))
    #self.bo = set_tensor(self.std_uniform_init(torch.empty(([self.hidden_dim,1]))))
    #self.by = set_tensor(self.std_uniform_init(torch.empty(([self.output_dim,1]))))
    self.bf = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bi = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bc = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.bo = set_tensor(torch.zeros((self.hidden_dim,1)))
    self.by = set_tensor(torch.zeros((self.output_dim,1)))
    self.zero_gradients()
    self.init_h = set_tensor(torch.empty([self.hidden_dim, self.batch_size]).normal_(mean=0.0,std=0.05))
    self.init_cell = set_tensor(torch.empty([self.hidden_dim, self.batch_size]).normal_(mean=0.0,std=0.05))
    self.embed = nn.Embedding(self.vocab_size, self.input_dim).to(DEVICE)

  def std_uniform_init(self,W):
    stdv = 1.0 / math.sqrt(self.hidden_dim)
    return init.uniform_(W, -stdv, stdv)

  def zero_gradients(self):
    #reset all gradients accumulated once the sequence is over
    #zero weights
    self.dWf = set_tensor(torch.zeros_like(self.Wf))
    self.dWi = set_tensor(torch.zeros_like(self.Wi))
    self.dWc = set_tensor(torch.zeros_like(self.Wc))
    self.dWo = set_tensor(torch.zeros_like(self.Wo))
    self.dWy = set_tensor(torch.zeros_like(self.Wy))
    #zero biases
    self.dbf = set_tensor(torch.zeros_like(self.bf))
    self.dbi = set_tensor(torch.zeros_like(self.bi))
    self.dbc = set_tensor(torch.zeros_like(self.bc))
    self.dbo = set_tensor(torch.zeros_like(self.bo))
    self.dby = set_tensor(torch.zeros_like(self.by))

  def copy_params_from(self, model):
    #copy weights
    self.dWf = set_tensor(model.dWf.clone())
    self.dWi = set_tensor(model.dWi.clone())
    self.dWc = set_tensor(model.dWc.clone())
    self.dWo = set_tensor(model.dWo.clone())
    self.dWy = set_tensor(model.dWy.clone())
    #copy biases
    self.dbf = set_tensor(model.dbf.clone())
    self.dbi = set_tensor(model.dbi.clone())
    self.dbc = set_tensor(model.dbc.clone())
    self.dbo = set_tensor(model.dbo.clone())
    self.dby = set_tensor(model.dby.clone())

  def cell_forward(self, inp, hprev, cellprev,t):
    #forward pass for a single timestep of the LSTM
    #inp = input at that timestep [Features x Batch]
    #hprev = previous hidden states [Hidden_size x Batch]
    #cellprev = previous cell state [Hidden_size x Batch]
    #t = position in sequence [int]
    #implements the forward pass of the LSTM. Saves the predictions for later inference
    #print("inp: ", inp.shape)
    #print("hprev: ",hprev.shape)
    #print("embed shape: ", self.embed(inp).shape)
    embed = self.embed(inp).permute(1,0)
    #print("EMBED: ", embed[:,0])
    #print("embed: ", embed.shape)
    #print("hprev: ", hprev.shape)
    z = torch.cat((embed, hprev),axis=0)
    # see if a linear embedding layer helps here!
    #z = self.embed @ z
    #print("z: ", z.shape)
    #forget gate
    self.mu_f_activations[t] = self.Wf @ z + self.bf
    #self.mu_f[t] = torch.sigmoid(self.mu_f_activations[t])
    self.mu_f[t] = F.sigmoid(self.mu_f_activations[t])
    #print("mu_f: ", self.mu_f[t])
    #input gate
    self.mu_i_activations[t] = self.Wi @ z + self.bi
    #self.mu_i[t] = torch.sigmoid(self.mu_i_activations[t])
    self.mu_i[t] = F.sigmoid(self.mu_i_activations[t])
    #print("mu_i: ", self.mu_i[t])
    #control gate
    self.mu_c_activations[t] = self.Wc @ z + self.bc
    #self.mu_c[t] = torch.tanh(self.mu_c_activations[t])
    self.mu_c[t] = F.tanh(self.mu_c_activations[t])
    #print("mu_c: ", self.mu_c[t])
    #output gate
    self.mu_o_activations[t] = self.Wo @ z + self.bo
    #self.mu_o[t] = torch.sigmoid(self.mu_o_activations[t])
    self.mu_o[t] =F.sigmoid(self.mu_o_activations[t])
    #print("mu_o: ", self.mu_o[t])

    self.mu_cell[t] = torch.mul(self.mu_f[t], cellprev) + torch.mul(self.mu_i[t], self.mu_c[t])
    #print("mu_cell: ", self.mu_cell[t])
    #self.mu_h[t] = torch.mul(self.mu_o[t], torch.tanh(self.mu_cell[t]))
    self.mu_h[t] = torch.mul(self.mu_o[t], torch.tanh(self.mu_cell[t]))
    #print("mu_h: ", self.mu_h[t])
    self.mu_y[t] = torch.sigmoid(self.Wy @ self.mu_h[t] + self.by)
    #print("mu_y: ", self.mu_y[t])
    return self.mu_y[t],self.mu_h[t], self.mu_cell[t]

  def cell_forward_old(self,inp,hprev,cellprev,t):
    # so effectively a linear layer. If this can't learn there is osmething REALLY DEEP going on here 
    #print("inp: ", inp.shape)
    z = torch.cat((inp, hprev),axis=0)
    #print("wf: ", self.Wf.shape)
    #print("z: ", z.shape)
    #print("bf: ", self.bf.shape)
    h = F.relu(self.Wf @ z + self.bf)
    self.mu_y[t] = self.Wy @ h + self.by
    #print("ypred!: ", self.mu_y[t].shape)
    return self.mu_y[t],hprev, cellprev


  def cell_backward(self,inp,true_labels,cellprev,hprev,dc_back, dh_back,t):
    # Returns the backwards gradients for a single LSTM cell timstep
    #takes as input: inputs [Features x Batch] , labels [1 x Batch], cellprev, hprev,dc_batck, dh_back: [Hidden_size x Batch], t = timestep in seqlen [int]
    z = torch.cat((inp, hprev),axis=0)
    #compute gradients    
    dy = true_labels - self.mu_y[t]
    dh = (self.Wy.T @ dy) + dh_back  #any activation function on the output?
    do = torch.mul(sigmoid_deriv(self.mu_o_activations[t]),torch.mul(tanh(self.mu_cell[t]), dh))
    dcell = (self.mu_o[t] * dh * tanh_deriv(self.mu_cell[t])) +dc_back
    df = torch.mul(sigmoid_deriv(self.mu_f_activations[t]), torch.mul(cellprev, dcell))
    di = torch.mul(sigmoid_deriv(self.mu_i_activations[t]),torch.mul(dcell, self.mu_c[t]))
    dc = torch.mul(tanh_deriv(self.mu_c_activations[t]),torch.mul(dcell,self.mu_i[t]))

    #I think these accumulation steps should all be + now. Although I need to check numerically
    #accumulate weights
    self.dWf += df @ z.T
    self.dWi += di @ z.T
    self.dWc += dc @ z.T
    self.dWo += do @ z.T
    self.dWy += dy @ self.mu_h[t].T


    self.dbf += torch.sum(df,dim=1,keepdim=True)
    self.dbi += torch.sum(di ,dim=1,keepdim=True)
    self.dbc += torch.sum(dc ,dim=1,keepdim=True)
    self.dbo += torch.sum(do ,dim=1,keepdim=True)
    self.dby +=  torch.sum(dy,dim=1,keepdim=True)

    #compute backwards derivatives
    dc_back = self.mu_f[t] * dcell 
    dh_back = (self.Wf.T @ df) +(self.Wi.T @ di) + (self.Wo.T @ do) + (self.Wc.T @ dc)
    return dc_back, dh_back


  def update_parameters(self):
    #update weights
    #self.Wf += 1 * self.learning_rate * torch.clamp(self.dWf,min=-5,max=5)
    #print("dwy: ", torch.clamp(self.dWy,min=-5,max=5))

    self.Wi += 1 * self.learning_rate * torch.clamp(self.dWi,min=-self.clamp_val,max=self.clamp_val)
    self.Wc += 1 * self.learning_rate * torch.clamp(self.dWc,min=-self.clamp_val,max=self.clamp_val)
    self.Wo += 1 * self.learning_rate * torch.clamp(self.dWo,min=-self.clamp_val,max=self.clamp_val)
    self.Wy += 1 * self.learning_rate * torch.clamp(self.dWy,min=-self.clamp_val,max=self.clamp_val)
    #update biases
    self.bf += self.learning_rate * torch.clamp(self.dbf,min=-self.clamp_val,max=self.clamp_val)
    self.bi += self.learning_rate * torch.clamp(self.dbi,min=-self.clamp_val,max=self.clamp_val)
    self.bc += self.learning_rate * torch.clamp(self.dbc,min=-self.clamp_val,max=self.clamp_val)
    self.bo += self.learning_rate * torch.clamp(self.dbo,min=-self.clamp_val,max=self.clamp_val)
    self.by += self.learning_rate * torch.clamp(self.dby,min=-self.clamp_val,max=self.clamp_val)
    #zero gradients
    self.zero_gradients()

  def initialize_caches(self,T):
    #reset the caches for the new sequence
    self.mu_f_activations = [[] for i in range(T)]
    self.mu_f = [[] for i in range(T)]
    self.mu_i_activations = [[] for i in range(T)]
    self.mu_i = [[] for i in range(T)]
    self.mu_c_activations = [[] for i in range(T)]
    self.mu_c = [[] for i in range(T)]
    self.mu_o_activations = [[] for i in range(T)]
    self.mu_o = [[] for i in range(T)]
    self.mu_cell = [[] for i in range(T)]
    self.mu_h = [[] for i in range(T)]
    self.mu_y = [[] for i in range(T)]
    self.hprev = [[] for i in range(T+1)]
    self.cellprev = [[] for i in range(T+1)]

  def forward(self,input_seq,init_h=None,init_cell = None):
    #loop over the sequence to do the first forward pass
    T = len(input_seq)
    self.initialize_caches(T)
    #initialize starting hprev and cellprev
    #I found that random initializations each time added too much noise for gradients to learn successfully
    #self.hprev[0] = init_h if init_h is not None else set_tensor(torch.empty([self.hidden_dim, self.batch_size]).normal_(mean=0,std=0.1))
    #self.cellprev[0] = init_cell if init_cell is not None else set_tensor(torch.empty([self.hidden_dim,self.batch_size]).normal_(mean=0, std=0.1))
    #set some FIXED noise here from the initialization. WIll this fix it?
    self.hprev[0] = init_h if init_h is not None else self.init_h
    self.cellprev[0] = init_cell if init_cell is not None else self.init_cell
    #roll forwards
    for (t, inp) in enumerate(input_seq):
      #roll forwards across the sequence
      out, self.hprev[t+1], self.cellprev[t+1] = self.cell_forward(inp, self.hprev[t], self.cellprev[t], t)
    return self.mu_y

  def backward(self, input_seq, target_seq):
    #this function loops over each element of the sequence backwards and performs backwards inference for each LSTM cell
    T = len(input_seq)
    assert T == len(target_seq), "Input and target sequence must be same length"
    #initialize first backwards gradients to 0
    self.dc_back = set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
    self.dh_back = set_tensor(torch.zeros([self.hidden_dim, self.batch_size]))
    #begin the backwards loop
    for (t, (inp, targ)) in reversed(list(enumerate(zip(input_seq,target_seq)))):
      self.dc_back, dh_back = self.cell_backward(inp, targ,self.cellprev[t],self.hprev[t],self.dc_back,self.dh_back,t)
      #print("in backward: ", dh_back.shape)
      self.dh_back = dh_back[self.input_dim:,:]
      #print("after: ", dh_back.shape)
      #print("after: ", self.dh_back.shape)
      #self.dh_back = dh_back[0:self.hidden_dim,:]
    return self.dc_back, self.dh_back,self.dWf, self.dWi, self.dWc,self.dWo,self.dWy

  def sample_sentence(self,input_char, n_steps,sample_char=False,init_h=None,init_cell=None,temp=20):
    input_seq = [set_tensor(torch.zeros_like(input_char)) for i in range(n_steps)]
    input_seq[0] = torch.tensor(input_char).reshape(1,)
    hprev = init_h if init_h is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    cellprev = init_cell if init_cell is not None else set_tensor(torch.zeros([self.hidden_dim,1]))
    output_str = ""
    #setup the network
    self.initialize_caches(n_steps)
    for n in range(1,n_steps):
      pred_y,hprev,cellprev = self.cell_forward(input_seq[n-1],hprev,cellprev,n)
      if sample_char:
       probs=F.softmax(pred_y.squeeze(1) * temp)
       cat = dist.Categorical(probs=probs)
       char_idx = cat.sample()
      else:
        #get the maximum char
        char_idx = torch.argmax(pred_y.squeeze(1))
        
      input_seq[n] = char_idx.reshape(1,)
      char = idx2char[char_idx]
      output_str+=char
    return output_str

  def save_model(self, logdir, savedir,losses=None, accs=None):
    np.save(logdir + "/Wf.npy", self.Wf.detach().cpu().numpy())
    np.save(logdir + "/Wi.npy", self.Wi.detach().cpu().numpy())
    np.save(logdir + "/Wc.npy", self.Wc.detach().cpu().numpy())
    np.save(logdir + "/Wo.npy", self.Wo.detach().cpu().numpy())
    np.save(logdir + "/Wy.npy", self.Wy.detach().cpu().numpy())

    np.save(logdir + "/bf.npy", self.bf.detach().cpu().numpy())
    np.save(logdir + "/bi.npy", self.bi.detach().cpu().numpy())
    np.save(logdir + "/bc.npy", self.bc.detach().cpu().numpy())
    np.save(logdir + "/bo.npy", self.bo.detach().cpu().numpy())
    np.save(logdir + "/by.npy", self.by.detach().cpu().numpy())

    np.save(logdir + "/init_h.npy", self.init_h.detach().cpu().numpy())
    np.save(logdir + "/init_cell.npy", self.init_cell.detach().cpu().numpy())
    #save all the embedding parameters
    embed_params = list(self.embed.parameters())
    for (i,p) in enumerate(embed_params):
        np.save(logdir + "/embed_"+str(i)+".npy",p.detach().cpu().numpy())
        
    #SAVE the results to the edinburgh computer from scratch space to main space
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
        Wf = np.load(save_dir+"/Wf.npy")
        self.Wf = set_tensor(torch.from_numpy(Wf))
        Wi = np.load(save_dir+"/Wi.npy")
        self.Wi = set_tensor(torch.from_numpy(Wi))
        Wc = np.load(save_dir+"/Wc.npy")
        self.Wc = set_tensor(torch.from_numpy(Wc))
        Wo = np.load(save_dir+"/Wo.npy")
        self.Wo = set_tensor(torch.from_numpy(Wo))
        Wy = np.load(save_dir+"/Wy.npy")
        self.Wy = set_tensor(torch.from_numpy(Wy))

        bf = np.load(save_dir+"/bf.npy")
        self.bf = set_tensor(torch.from_numpy(bf))
        bi = np.load(save_dir+"/bi.npy")
        self.bi = set_tensor(torch.from_numpy(bi))
        bc = np.load(save_dir+"/bc.npy")
        self.bc = set_tensor(torch.from_numpy(bc))
        bo = np.load(save_dir+"/bo.npy")
        self.bo = set_tensor(torch.from_numpy(bo))
        by = np.load(save_dir+"/by.npy")
        self.by = set_tensor(torch.from_numpy(by))

        init_h = np.load(save_dir + "/init_h.npy")
        self.init_h = set_tensor(torch.from_numpy(init_h))
        init_cell = np.load(save_dir + "/init_cell.npy")
        self.init_cell = set_tensor(torch.from_numpy(init_cell))

        embed = np.load(save_dir +"/embed_0.npy")
        self.embed.weight = nn.Parameter(set_tensor(torch.from_numpy(embed)))

  def net_set_parameters(self):
    #weight parameters
    self.Wf = nn.Parameter(self.Wf)
    self.Wi = nn.Parameter(self.Wi)
    self.Wc = nn.Parameter(self.Wc)
    self.Wo = nn.Parameter(self.Wo)
    self.Wy = nn.Parameter(self.Wy)
    #bias parameters
    self.bf = nn.Parameter(self.bf)
    self.bi = nn.Parameter(self.bi)
    self.bc = nn.Parameter(self.bc)
    self.bo = nn.Parameter(self.bo)
    self.by = nn.Parameter(self.by)
    #self.embed = nn.Parameter(self.embed)
    #learn initial weights?!
    self.init_h = nn.Parameter(self.init_h)
    self.init_cell = nn.Parameter(self.init_cell)

  def train(self, dataset,n_epochs,logdir, savedir,old_savedir="",init_embed_path = "None",save_every=20):
    if old_savedir != "None":
        self.load_model(old_savedir)
    self.net_set_parameters()
    params = [self.Wf,self.Wi,self.Wc,self.Wo,self.Wy,self.bf,self.bi,self.bc,self.bo,self.by]
    #params += list(self.embed.parameters())
    #0.0005
    optimizer = optim.SGD(params, lr=self.learning_rate)
    losses = []
    accs = []
    #create a file to write the outputs to
    output_file = open(logdir +"/output.txt", "w")
    for n in range(n_epochs):
      print("Epoch ", n)
      for (i,(input_seq, target_seq)) in enumerate(dataset):
        input_seq = list(torch.tensor(torch.from_numpy(input_seq.numpy()),dtype=torch.long).permute(1,0).to(DEVICE))
        target_seq = list(set_tensor(torch.from_numpy(onehot(target_seq, self.vocab_size)).float().permute(2,1,0)))
        pred_ys = self.forward(input_seq)
        optimizer.zero_grad()
        L = 0.0
        for pred_y,targ in zip(pred_ys,target_seq):
          L += torch.sum((pred_y - targ)**2)
        print("Loss: ", L)
        acc = sequence_accuracy(self,target_seq)
        print("Accuracy: ", acc)
        L.backward() 
        optimizer.step()
        #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char = True)))
        #print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char=False)))
        #only save after each n goes to save a bit of space wil still get PLENTY of results
        if i % save_every == 0:
            losses.append(L.item())
            accs.append(acc)
            print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char = True)),file=output_file)
            print("SAMPLED TEXT : " + str(self.sample_sentence(input_seq[0][int(np.random.uniform(low=0,high=self.batch_size))],len(input_seq),sample_char=False)),file=output_file)
        if i % 200 == 0:
            #save model after each epoch
            print("FINISHED EPOCH: " + str(n) + " SAVING MODEL")
            self.save_model(logdir, savedir,losses,accs)

print("reached end of file before")

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
    parser.add_argument("--sample_every",type=int,default=200)
    parser.add_argument("--network_type",type=str,default="backprop")
    parser.add_argument("--sample_char",type=boolcheck,default="True")
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--init_embed_path",type=str,default="None") #"/home/s1686853/lstm_backprop_experiments/backprop_baseline_run5/0/embed_0.npy")

    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    dataset, vocab_size,char2idx,idx2char = get_lstm_dataset(args.seq_len, args.batch_size)

    input_size = vocab_size
    hidden_size = args.hidden_size
    output_size = vocab_size
    batch_size = args.batch_size
    inference_learning_rate = args.inference_learning_rate
    weight_learning_rate = args.weight_learning_rate
    n_inference_steps = args.n_inference_steps
    n_epochs = args.N_epochs
    save_every = args.save_every
    sample_every = args.sample_every

    #define networks
    if args.network_type == "pc":
        net = PC_LSTM(input_size, hidden_size,output_size,vocab_size,batch_size,inference_learning_rate,weight_learning_rate/2,n_inference_steps)
    elif args.network_type == "backprop":
        net = Backprop_LSTM(input_size,hidden_size,output_size,vocab_size,batch_size,weight_learning_rate)
    else:
        raise Exception("Unknown network type entered")

    #train!
    net.train(dataset, int(n_epochs),args.logdir, args.savedir,old_savedir=args.old_savedir,init_embed_path = args.init_embed_path,save_every=args.save_every)

