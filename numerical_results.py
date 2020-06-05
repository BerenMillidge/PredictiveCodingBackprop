#Various utilities and experiments for testing that the numerical solution works for a branching nonlinear function
# This code should reproduce figures in section 5.1 -- numerical results
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns

### utility functions ###
def get_printer(msg):
    #this function is used by register hook in that it's a function which can print info about the gradient from inside the computation graph.
    def printer(tensor):
        #print("tensor: grad: ", tensor.grad)
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.max()} min: {tensor.min()}"
                  f" mean: {tensor.mean()}")
    return printer

def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))

### Gradient functions ###
def pytorch_gradient_descent():
    #check that naive gradient descent done with true pytorch gradients works
    y0 = nn.Parameter(torch.tensor([5],dtype=torch.float32))
    theta = nn.Parameter(torch.tensor([2],dtype=torch.float32))
    for i in range(50):
        y1 = y0 * theta
        y2 = y1.pow(1/2)
        y3 = y0.pow(2)
        y4 = torch.tan(y2) + torch.sin(y3)
        T = 1
        print("Y$: ", y4.item())
        L = (T-y4).pow(2)
        print("LOSS: ", L.item())
        L.backward()
        print("Grad: ",y0.grad.item())
        y0 = nn.Parameter(torch.tensor([y0.data - (0.01 * y0.grad)],dtype=torch.float32))
        #update step
        #y0.data -= 0.0001 * y0.grad

def get_pytorch_gradients():
    #correct gradients using pytorch's autodiff
    y0 = nn.Parameter(torch.tensor([5],dtype=torch.float32))
    theta = nn.Parameter(torch.tensor([2],dtype=torch.float32))
    y1 = y0 * theta
    register_hook(y1, "y1")
    print("Y1: ",y1)
    y2 = y1.pow(1/2)
    print("Y2: ",y2)
    register_hook(y2, "y2")
    y3 = y0.pow(2)
    print("Y3: ",y3)
    register_hook(y3, "y3")
    y4 = torch.tan(y2) + torch.sin(y3)
    register_hook(y4, "y4")
    T = 1
    print("Y4: ", y4.item())
    L = 0.5 *(T-y4).pow(2)
    print("LOSS: ", L.item())
    L.backward()
    print("Grad: ",y0.grad.item())
    print("y1 grad: ",y1.grad)
    print("y2 grad: ", y2.grad)
    print("y3 grad: ", y3.grad)
    print("y4 grad: ", y4.grad)
    y0_true_grad = y0.grad.clone()
    y1_true_grad = y1.grad.clone()
    y2_true_grad = y2.grad.clone()
    y3_true_grad = y3.grad.clone()
    y4_true_grad = y4.grad.clone()
    return y0_true_grad,y1_true_grad,y2_true_grad,y3_true_grad,y4_true_grad

def f(y0):
    #This implements the forward pass of the arbitrary nonlinear function
    theta = nn.Parameter(torch.tensor([2],dtype=torch.float32))
    y1 = y0 * theta
    y2 = y1.pow(1/2)
    y3 = y0.pow(2)
    y4 = torch.tan(y2) + torch.sin(y3)
    T = 3
    print("Y4: ", y4.item())
    L = 0.5 * (T-y4).pow(2)
    return L

def finite_diffs(f, y0, delta=1e-4):
    #take the finite difference gradient (first order)
    return (f(y0+delta) - f(y0)) / delta

def finite_differences_gradient(delta=1e-4):
    #take gradients via finite differences
    #Finite differences gradient
    #only gives gradient wrt output.
    y0 = nn.Parameter(torch.tensor([5],dtype=torch.float32))
    finite_diff = finite_diffs(f, y0,delta)
    print("Finite difference gradient at output: ", finite_diff)
    return finite_diff


def predictive_coding_gradient(y0_true_grad,learning_rate=0.02,n_inference_steps=500):
    e_y0s = []
    with torch.no_grad():
        y0 = torch.tensor([5],dtype=torch.float32)
        theta = torch.tensor([2],dtype=torch.float32)
        y1 = y0 * theta
        y2 = y1.pow(1/2)
        y3 = y0.pow(2)
        y4 = torch.tan(y2) + torch.sin(y3)
        mu1 = y0 * theta
        mu2 = y1.pow(1/2)
        mu3 = y0.pow(2)
        mu4 = torch.tan(y2) + torch.sin(y3)
        T = 1
        print("Y4: ", y4.item())
        L = (T-mu4).pow(2)
        e4 = (T-mu4)
        for i in range(n_inference_steps):
            e4 = 1 * (T - mu4)
            e3 = y3 - mu3
            e2 = y2 - mu2
            e1 = y1 - mu1
            e0 = -(e1 * theta) - (2*e3*y0)
            y0dot =  (-(e1 * theta) -(2*e3*y0))
            y1dot = e1 - (0.5 * e2 * mu1.pow(-0.5))
            y2dot = e2 -(e4 * 1/(torch.cos(mu2).pow(2)))
            y3dot = e3 -(e4 * torch.cos(mu3))
            #print("GRADIENT e0: ", e0.item())
            e_y0s.append(e0.item() - y0_true_grad.item())
            print("grad divergence: ", e0.item() - y0_true_grad.item())
            y1 -= (learning_rate * y1dot)
            y2 -= (learning_rate * y2dot)
            y3 -= (learning_rate * y3dot)
            total_sum = e4.pow(2)+e3.pow(2)+e2.pow(2)+e1.pow(2)
            print("total sum: ",total_sum)
    return e_y0s

### plotting config ###
TICK_SIZE = 14
LEGEND_SIZE = 14
LABEL_SIZE = 14
FIG_SIZE = [4, 6]
plt.rc("xtick", labelsize=TICK_SIZE)
plt.rc("ytick", labelsize=TICK_SIZE)
plt.rc("legend", fontsize=LEGEND_SIZE)

def get_color_palette():
    palette = sns.color_palette("Paired", 12)
    _colors = [palette[5], palette[7], palette[0], palette[2]]
    return _colors

colors = get_color_palette()

plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

### Plotting Functions ###

def plot_numerical_divergence(e_y0s):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Divergence from True Gradient",fontsize=20,fontweight="bold",pad=25)
    ax.plot(e_y0s,label="Prediction error")
    plt.xlabel("Variational Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("Mean Divergence",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend()
    legend.fontsize=18
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_divergence.jpg")
    plt.show()

def plot_log_divergence(log_ey0s):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Log Prediction Error",fontsize=20,fontweight="bold",pad=25)
    ax.plot(log_ey0s,label="Log Prediction Error")
    plt.xlabel("Variational Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("Log Divergence",fontsize=20,style="oblique",labelpad=10)
    plt.yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend()
    legend.fontsize=18
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_log_divergence.jpg")
    plt.show()

def learning_rate_comparison(lrs):
    y0_true_grad,y1_true_grad,y2_true_grad,y3_true_grad,y4_true_grad = get_pytorch_gradients()
    ey0s_list = []
    for lr in lrs:
        ey0s = predictive_coding_gradient(y0_true_grad,learning_rate=lr)
        ey0s_list.append(ey0s)
    return ey0s_list

def inference_steps_comparison(steps):
    y0_true_grad,y1_true_grad,y2_true_grad,y3_true_grad,y4_true_grad = get_pytorch_gradients()
    ey0s_list = []
    for step in steps:
        ey0s = predictive_coding_gradient(y0_true_grad,n_inference_steps=step)
        ey0s_list.append(ey0s)
    return ey0s_list


def plot_learning_rate_comparison(ey0s_list, lrs,log_scale=False):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Learning Rate Comparison",fontsize=20,fontweight="bold",pad=25)
    for (ey0,lr) in zip(ey0s_list,lrs):
        labelstr = "Learning Rate " + str(lr)
        ax.plot(ey0,label=labelstr)
    plt.xlabel("Variational Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("Mean Divergence",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    log = ""
    if log_scale:
        plt.yscale("log")
        log = "_log"
    legend = plt.legend()
    legend.fontsize=16
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_learning_rate_comparison" + str(log)+".jpg")
    plt.show()

def plot_inference_steps_comparison(ey0s_list, steps,log_scale=False):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Number of Iterations",fontsize=16,fontweight="bold",pad=25)
    for (ey0,step) in zip(ey0s_list,steps):
        labelstr =  str(step) + "Inference Steps"
        ax.plot(ey0,label=labelstr)
    plt.xlabel("Variational iteration",fontsize=16,style="oblique",labelpad=10)
    plt.ylabel("Total Divergence from true gradient",fontsize=16,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    log = ""
    if log_scale:
        plt.yscale("log")
        log = "_log"
    legend = plt.legend()
    legend.fontsize=10
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_learning_rate_comparison" + str(log)+".jpg")
    plt.show()


if __name__ =='__main__':
    y0_true_grad,y1_true_grad,y2_true_grad,y3_true_grad,y4_true_grad = get_pytorch_gradients()
    ey0s = predictive_coding_gradient(y0_true_grad)
    plot_numerical_divergence(ey0s)
    plot_log_divergence(ey0s)
    learning_rates = [0.01,0.02,0.05,0.1,0.2,0.5,1]
    ey0s_list = learning_rate_comparison(learning_rates)
    plot_learning_rate_comparison(ey0s_list,learning_rates)
