import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import requests 
import glob
import zipfile
import unicodedata
import string
import random
from utils import *
import math
import subprocess
import argparse
from datetime import datetime

subprocess.call(['echo','opening file'])

def download_extract_names_data():
  url = "https://download.pytorch.org/tutorial/data.zip"
  r = requests.get(url, allow_redirects=True)

  open('data.zip', 'wb').write(r.content)
  with zipfile.ZipFile("data.zip","r") as zip_ref:
      zip_ref.extractall("data")

def find_files(path):
  return glob.glob(path)

#download_extract_names_data()
#print(find_files("data/data/names/*.txt"))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#turn a unicode string into ascii
def to_ascii(s):
  return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(to_ascii('Ślusàrski'))


def read_lines(filename):
  lines = open(filename, encoding='utf-8').read().strip().split('\n')
  return [to_ascii(line) for line in lines]

def files_to_categories(filelist):
  category_lines  = {}
  all_categories = []
  for filename in filelist:
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category]= lines
    n_categories = len(all_categories)
  return category_lines, all_categories, n_categories


filelist = find_files("./data/data/names/*.txt")
category_lines, all_categories, n_categories = files_to_categories(filelist)
print(category_lines["Italian"][:5])
subprocess.call(['echo','files downloaded'])

def char2idx(char):
  return all_letters.find(char)

def char2tensor(char):
  tensor = torch.zeros(1, n_letters)
  tensor[0][char2idx(char)] = 1
  return tensor

def line2tensor(line):
  tensor = torch.zeros(len(line),1,n_letters)
  for li, letter in enumerate(line):
    tensor[li][0][char2idx(letter)] = 1
  return tensor

print(char2tensor("J"))
print(line2tensor("Bibblebob"))



def category_from_output(output):
  top_cat = torch.argmax(output).item()
  return all_categories[top_cat],top_cat

def random_choice(l):
  return l[random.randint(0,len(l)-1)]

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def random_training_example():
  category = random_choice(all_categories)
  line = random_choice(category_lines[category])
  category_tensor= categoryTensor(category)
  line_tensor = line2tensor(line)
  return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_example()
    print('category =', category, '/ line =', line)
    print("category tensor = ", category_tensor.shape)
    print("line_tensor = ", line_tensor.shape )


class PC_RNN(object):
  def __init__(self, hidden_size, input_size, output_size,batch_size, fn, fn_deriv,inference_learning_rate, weight_learning_rate, n_inference_steps,device="cpu"):
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
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
      if i == len(input_seq)-1:
        self.y_preds = linear(self.Wy @ self.h_preds[i+1])
    return self.y_preds

  def infer(self, input_seq, targ,fixed_predictions=True):
    with torch.no_grad():
      #input sequence = [list of [Batch_size x Feature_Dimension]] seq len 
      #self.e_ys = [[] for i in range(len(target_seq))] #ouptut prediction errors
      self.e_hs = [[] for i in range(len(input_seq))] # hidden state prediction errors
      # test order of for loops -- i.e. iterate each iteration sweep through the whole RNN or 
      for i, inp in reversed(list(enumerate(input_seq))):
        #print("Inference step: ", n)
        #hdelta_sum = 0
        for n in range(self.n_inference_steps):
          if i == len(input_seq)-1:
            self.e_ys =  targ - self.y_preds #if targ is not None else None
          #hs[i+1] = current hidden state -- hs[i] = past time step
          if fixed_predictions == False:
            self.h_preds[i+1] = self.fn(self.Wh @ self.hs[i] + self.Wx @ inp)
          self.e_hs[i] = self.hs[i+1] - self.h_preds[i+1]
          hdelta = self.e_hs[i].clone()
          #if self.e_ys[i] is not None:
          #hdelta -= self.Wy.T @ (self.e_ys[i] * linear_deriv(self.Wy @ self.hs[i+1]))
          if i == len(input_seq) -1:
            hdelta -= self.Wy.T @ (self.e_ys * linear_deriv(self.Wy @ self.h_preds[i+1]))
          if i < len(input_seq)-1:
            #fn_deriv =  self.fn_deriv(self.Wh @ self.hs[i] + self.Wx @ inp)
            fn_deriv =  self.fn_deriv(self.Wh @ self.h_preds[i+1] + self.Wx @ input_seq[i+1])
            hdelta -= self.Wh.T @ (self.e_hs[i+1] * fn_deriv)
          self.hs[i+1] -= self.inference_learning_rate * hdelta
          if fixed_predictions == False:
            self.y_preds = linear(self.Wy @ self.hs[i+1])
      return self.e_ys, self.e_hs

  def update_weights(self, input_seq,update_weights=True):
    with torch.no_grad():
      dWy = set_tensor(torch.zeros_like(self.Wy))
      dWx = set_tensor(torch.zeros_like(self.Wx))
      dWh = set_tensor(torch.zeros_like(self.Wh))
      # go back in reverse through the graph and sum up everything
      for i in reversed(list(range(len(input_seq)))):
        #fn_deriv = self.fn_deriv(self.Wh @ self.hs[i] + self.Wx @ input_seq[i])
        #dWy += (self.e_ys[i] * linear_deriv(self.Wy @ self.hs[i+1])) @ self.hs[i+1].T #if self.e_ys[i] is not None else torch.zeros_like(self.Wy)
        #dWx += (self.e_hs[i] * fn_deriv) @ input_seq[i].T
        #dWh += (self.e_hs[i] * fn_deriv) @ self.hs[i].T
        #print("in update weights: ",i)
        fn_deriv = self.fn_deriv(self.Wh @ self.h_preds[i] + (self.Wx @ input_seq[i]))
        if i == len(input_seq)-1:
          dWy += (self.e_ys * linear_deriv(self.Wy @ self.h_preds[i+1])) @ self.h_preds[i+1].T #if self.e_ys[i] is not None else torch.zeros_like(self.Wy)
        dWx += (self.e_hs[i] * fn_deriv) @ input_seq[i].T
        dWh += (self.e_hs[i] * fn_deriv) @ self.h_preds[i].T
      if update_weights:
        self.Wy += self.weight_learning_rate * torch.clamp(dWy,-self.clamp_val,self.clamp_val)
        self.Wx += self.weight_learning_rate * torch.clamp(dWx,-self.clamp_val, self.clamp_val)
        self.Wh += self.weight_learning_rate * torch.clamp(dWh,-self.clamp_val, self.clamp_val)
      return dWy, dWx, dWh

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

  def train(self, n_epochs,logdir,savedir,old_savedir="None",save_every=50):
    if old_savedir != "None":
        self.load_model(old_savedir)
    with torch.no_grad():
      acc = 0
      loss = 0
      losses = []
      accs = []
      for n in range(n_epochs):
        category, line, category_tensor, line_tensor = random_training_example()
        input_seq = [set_tensor(line_tensor[i,:,:].permute(1,0)) for i in range(len(line_tensor))]
        target = set_tensor(category_tensor.permute(1,0))
        ypreds = self.forward_sweep(input_seq)
        self.infer(input_seq, target)
        self.update_weights(input_seq)
        loss += torch.sum((target - ypreds)**2).item()
        if torch.argmax(target) == torch.argmax(ypreds):
          acc +=1
        if n % save_every == 0:
          print("Epoch: ",n)
          print("Loss: ", loss/save_every)
          print("acc: ", acc/save_every)
          losses.append(loss)
          accs.append(acc)
          loss = 0
          acc = 0
        if n % 200 == 0:
          self.save_model(logdir,savedir, losses,accs)


# let's generate a comparison backprop RNN so I can have the analytiacl weight updates done here which would be nice and straightforward to do
# so what is the ultimate goal here? it's really hard to tell. let's focus and get the weight update done, which is very important overall to be able to figure out
class Backprop_RNN(object):
  def __init__(self, hidden_size, input_size, output_size,batch_size, fn, fn_deriv,learning_rate):
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
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
    self.hs[0] = self.h0
    for i,inp in enumerate(input_seq):
      self.hs[i+1] = self.fn(self.Wh @ self.hs[i] + self.Wx @ inp)
      if i == len(input_seq) -1:
        self.y_preds = linear(self.Wy @ self.hs[i+1])
    return self.y_preds

  def backward_sweep(self,input_seq, target):
    self.dhs = [[] for i in range(len(input_seq)+1)]
    for i, inp in reversed(list(enumerate(input_seq))):
      dhdh = set_tensor(torch.zeros_like(self.hs[0]))
      if i == len(input_seq)-1:
        self.dys = target - self.y_preds
        dhdh += self.Wy.T @ (self.dys * linear_deriv(self.Wy @ self.hs[i+1]))
      if i < len(input_seq) -1:
        fn_deriv =  self.fn_deriv(self.Wh @ self.hs[i+1] + self.Wx @ input_seq[i+1])
        dhdh += self.Wh.T @ (self.dhs[i+1] * fn_deriv)
      #print(self.dhs[i])
      self.dhs[i]= dhdh
    return self.dhs, self.dys

  def update_weights(self,input_seq,update_weights=True):
    dWy = torch.zeros_like(self.Wy)
    dWx = torch.zeros_like(self.Wx)
    dWh = torch.zeros_like(self.Wh)
    for i,inp in reversed(list(enumerate(input_seq))):
      fn_deriv = self.fn_deriv(self.Wh @ self.hs[i] + self.Wx @ input_seq[i])
      if i == len(input_seq) -1:
        dWy += (self.dys * linear_deriv(self.Wy @ self.hs[i+1])) @ self.hs[i+1].T
      dWx += (self.dhs[i] * fn_deriv) @ inp.T
      dWh += (self.dhs[i] * fn_deriv) @ self.hs[i].T
    if update_weights:
      #2x since gradients are half in the 1/2 (x-t)^2 term
      self.Wy += self.learning_rate * torch.clamp(dWy,-self.clamp_val,self.clamp_val)
      self.Wx += self.learning_rate * torch.clamp(dWx,-self.clamp_val, self.clamp_val)
      self.Wh += self.learning_rate * torch.clamp(dWh,-self.clamp_val, self.clamp_val)
    return dWy, dWx, dWh

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

  def train(self, n_epochs,logdir,savedir,old_savedir="None",save_every=50):
    if old_savedir != "None":
        self.load_model(old_savedir)
    with torch.no_grad():
      acc = 0
      loss = 0
      losses = []
      accs = []
      for n in range(n_epochs):
        category, line, category_tensor, line_tensor = random_training_example()
        input_seq = [set_tensor(line_tensor[i,:,:].permute(1,0)) for i in range(len(line_tensor))]
        target = set_tensor(category_tensor.permute(1,0))
        ypreds = self.forward_sweep(input_seq)
        self.backward_sweep(input_seq, target)
        self.update_weights(input_seq)
        loss += torch.sum((target - ypreds)**2).item()
        if torch.argmax(target) == torch.argmax(ypreds):
          acc +=1
        if n % save_every == 0:
          print("Epoch: ",n)
          print("Loss: ", loss/save_every)
          print("acc: ", acc/save_every)
          losses.append(loss)
          accs.append(acc)
          loss = 0
          acc = 0
        if n % 200 == 0:
          self.save_model(logdir,savedir, losses,accs)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    subprocess.call(['echo', 'Initialized'])
        #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=1)
    parser.add_argument("--hidden_size",type=int,default=256)
    parser.add_argument("--n_inference_steps",type=int, default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--weight_learning_rate",type=float,default=0.0001)
    parser.add_argument("--N_epochs",type=int, default=100000)
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

    input_size = n_letters
    hidden_size = args.hidden_size
    output_size = n_categories
    batch_size = args.batch_size
    inference_learning_rate = args.inference_learning_rate
    weight_learning_rate = args.weight_learning_rate
    n_inference_steps = args.n_inference_steps
    n_epochs = args.N_epochs
    save_every = args.save_every

    #define networks
    if args.network_type == "pc":
        net = PC_RNN(hidden_size, input_size,output_size,batch_size,tanh, tanh_deriv,inference_learning_rate,weight_learning_rate,n_inference_steps)
    elif args.network_type == "backprop":
        net = Backprop_RNN(hidden_size,input_size,output_size,batch_size,tanh, tanh_deriv,weight_learning_rate)
    else:
        raise Exception("Unknown network type entered")

    #train!
    subprocess.call(['echo','beginning training'])
    net.train(int(n_epochs),args.logdir, args.savedir,old_savedir=args.old_savedir,save_every=args.save_every)



