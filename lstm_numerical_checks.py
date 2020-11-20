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
import numpy as np
from copy import deepcopy
from lstm import *
from utils import *

def net_set_parameters(net):
      #weight parameters
  net.Wf = nn.Parameter(net.Wf)
  net.Wi = nn.Parameter(net.Wi)
  net.Wc = nn.Parameter(net.Wc)
  net.Wo = nn.Parameter(net.Wo)
  net.Wy = nn.Parameter(net.Wy)
  #bias parameters
  net.bf = nn.Parameter(net.bf)
  net.bi = nn.Parameter(net.bi)
  net.bc = nn.Parameter(net.bc)
  net.bo = nn.Parameter(net.bo)
  net.by = nn.Parameter(net.by)
  return net

def get_bias_gradients(net):
  return net.dbf,net.dbi,net.dbc,net.dbo,net.dby

def get_weight_gradients(net):
  return net.dWf,net.dWi,net.dWc,net.dWo,net.dWy

def numerical_check(dataset,vocab_size, batch_size,n_inference_steps = 150):
  #this checks the numerical correctness of the accumulated gradients
  input_size = vocab_size
  hidden_size = 65
  output_size = vocab_size
  learning_rate = 0.1
  #net = PC_LSTM(input_size, hidden_size, output_size, batch_size, learning_rate,learning_rate, n_inference_steps)
  #net = net_set_parameters(net)
  inference_learning_rate = 0.1
  weight_learning_rate = 0.0001
  print("N inference steps: ", n_inference_steps)
  net = PC_LSTM(input_size, hidden_size,output_size,vocab_size,batch_size,inference_learning_rate,weight_learning_rate/2,n_inference_steps)
  backprop_net = Backprop_LSTM(input_size,hidden_size,output_size,vocab_size,batch_size,weight_learning_rate)
  net = net_set_parameters(net)
  init_h = nn.Parameter(set_tensor(torch.zeros([net.hidden_dim, net.batch_size])))
  init_c = nn.Parameter(set_tensor(torch.zeros([net.hidden_dim, net.batch_size])))


  for (i, (input_seq, target_seq)) in enumerate(dataset):
    input_seq = list(torch.tensor(torch.from_numpy(input_seq.numpy()),dtype=torch.long).permute(1,0).to(DEVICE))
    target_seq = list(set_tensor(torch.from_numpy(onehot(target_seq, vocab_size)).float().permute(2,1,0)))
    pred_ys = net.forward(input_seq,init_h =init_h,init_cell=init_c)
    #compute the real losses via autograd
    L = 0
    for pred_y,targ in zip(pred_ys,target_seq):
      L += torch.sum((pred_y - targ)**2)
    #backwards it
    print("Loss: ", L)
    L.backward()
    #get a list of gradients
    true_Wf_grad = net.Wf.grad.clone()
    true_Wi_grad = net.Wi.grad.clone()
    true_Wc_grad = net.Wc.grad.clone()
    true_Wo_grad = net.Wo.grad.clone()
    true_Wy_grad = net.Wy.grad.clone()
    
    true_bf_grad = net.bf.grad.clone()
    true_bi_grad = net.bi.grad.clone()
    true_bc_grad = net.bc.grad.clone()
    true_bo_grad = net.bo.grad.clone()
    true_by_grad = net.by.grad.clone()

    true_hback_grad = init_h.grad.clone()
    true_cback_grad = init_c.grad.clone()

    #print true gradients
    print("True dWf: ", true_Wf_grad[0:10,0])
    print("True dWi: ", true_Wi_grad[0:10,0])
    print("True dWc: ", true_Wc_grad[0:10,0])
    print("True dWo: ", true_Wo_grad[0:10,0])
    print("True dWy: ", true_Wy_grad[0:10,0])

    #print("True dbf: ", true_bf_grad[0:10,0])
    #print("True dbi: ", true_bi_grad[0:10,0])
    #print("True dbc: ", true_bc_grad[0:10,0])
    #print("True dbo: ", true_bo_grad[0:10,0])
    #print("True dby: ", true_by_grad[0:10,0])

    print("True hback: ", true_hback_grad[0:10,0])
    print("True cback: ", true_cback_grad[0:10,0])
    print(true_hback_grad.shape)

    # Backprop net gradients


    #PC net gradients
    PC_dc_back, PC_dh_back,PC_dWf, PC_dWi, PC_dWc,PC_dWo,PC_dWy = net.backward(input_seq, target_seq)
    print("PC dWf: ", PC_dWf[0:10,0] * 2)
    print("PC dWi: ", PC_dWi[0:10,0] * 2)
    print("PC dWc: ", PC_dWc[0:10,0] * 2)
    print("PC dWo: ", PC_dWo[0:10,0] * 2)
    print("PC dWy: ", PC_dWy[0:10,0] * 2)

    PC_dbf,PC_dbi,PC_dbc,PC_dbo,PC_dby = get_bias_gradients(net)
    #print("PC dbf: ", PC_dbf[0:10,0] * 2)
    #print("PC dbi: ", PC_dbi[0:10,0] * 2)
    #print("PC dbc: ", PC_dbc[0:10,0] * 2)
    #print("PC dbo: ", PC_dbo[0:10,0] * 2)
    ##print("PC dby: ", PC_dby[0:10,0] * 2)

    print("PC hback: ", PC_dh_back[0:10,0] * 2)
    print("PC cback: ", PC_dc_back[0:10,0] * 2)
    print(PC_dh_back.shape)
    return true_Wf_grad, true_Wi_grad, true_Wc_grad, true_Wo_grad, true_Wy_grad, PC_dWf * 2, PC_dWi * 2, PC_dWc * 2, PC_dWo * 2, PC_dWy * 2

    

def seq_len_divergence_comparison(seq_lens, batch_size=64):
      seq_len_divergences = []
  for i, seq_len in enumerate(seq_lens):
    print("Seq len: ", seq_len)
    dataset, vocab_size,char2idx,idx2char = get_lstm_dataset(seq_len, batch_size)
    true_Wf_grad, true_Wi_grad, true_Wc_grad, true_Wo_grad, true_Wy_grad, PC_dWf, PC_dWi, PC_dWc, PC_dWo, PC_dWy  = numerical_check_2(dataset,vocab_size,batch_size)
    divergence = 0
    divergence += torch.sum(torch.square(true_Wf_grad - PC_dWf)).item()
    divergence += torch.sum(torch.square(true_Wi_grad - PC_dWi)).item()
    divergence += torch.sum(torch.square(true_Wc_grad - PC_dWc)).item()
    divergence += torch.sum(torch.square(true_Wo_grad - PC_dWo)).item()
    divergence += torch.sum(torch.square(true_Wy_grad - PC_dWy)).item()
    seq_len_divergences.append(divergence)
  # plot
  fig = plt.figure()
  plt.title("Divergence against sequence length")
  plt.xlabel("Sequence length")
  plt.ylabel("Gradient Divergence")
  fig.savefig("seq_len_divergence_comparison.jpg")
  plt.plot(seq_lens, seq_len_divergences)
  return fig

def convergence_num_iterations_comparison(seq_lens, threshold, batch_size = 64,inference_steps_increment = 5):
  iterations_to_convergence = []
  for i, seq_len in enumerate(seq_lens):
    print("Seq len: ", seq_len)
    dataset, vocab_size,char2idx,idx2char = get_lstm_dataset(seq_len, batch_size)
    divergence = 1e10
    num_iterations = 0
    print("reaching threshold: ", divergence <= threshold)
    while divergence >= threshold:
      print("inside run numerical check")
      true_Wf_grad, true_Wi_grad, true_Wc_grad, true_Wo_grad, true_Wy_grad, PC_dWf, PC_dWi, PC_dWc, PC_dWo, PC_dWy  = numerical_check_2(dataset,vocab_size,batch_size,n_inference_steps = num_iterations + inference_steps_increment)
      divergence = 0
      divergence += torch.sum(torch.square(true_Wf_grad - PC_dWf)).item()
      divergence += torch.sum(torch.square(true_Wi_grad - PC_dWi)).item()
      divergence += torch.sum(torch.square(true_Wc_grad - PC_dWc)).item()
      divergence += torch.sum(torch.square(true_Wo_grad - PC_dWo)).item()
      divergence += torch.sum(torch.square(true_Wy_grad - PC_dWy)).item()
      print("divergence, ", divergence)
      num_iterations +=inference_steps_increment
    iterations_to_convergence.append(num_iterations)
  # plot
  fig = plt.figure()
  plt.title("Numerical iterations to convergence against sequence length")
  plt.ylabel("Iterations before convergence")
  plt.xlabel("Sequence Length")
  plt.plot(seq_lens, iterations_to_convergence)
  fig.savefig("convergence_num_iterations_comparison.jpg")
  plt.show()
  return fig


if __name__ == "__main__":
    seq_lens = [1,2,5,10,20,30,50,70,100,150,200]
    seq_len_divergence_comparison(seq_lens)
    convergence_num_iterations_comparison(seq_lens, 0.001)
