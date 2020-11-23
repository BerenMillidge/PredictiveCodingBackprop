import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

### Initial Styles ###
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

def get_average_reward(path,n_seeds):
    first_path = path + "/0/metrics.json"
    reward = np.array(load_json(first_path)["test_rewards"],dtype=np.float64)#[0:70]
    for i in range(1,n_seeds):
        i_path = path + "/" + str(i) + "/metrics.json"
        rewards = np.array(load_json(i_path)["test_rewards"])#[0:70]
        reward += rewards
    return reward / n_seeds

print("loading...")
pc_path = sys.argv[1]
backprop_path = sys.argv[2]
title = str(sys.argv[3])
EPOCH_NUM = 47

def get_results(basepath,cnn=True,merged=False):
    ### Loads results losses and accuracies files ###
    dirs = os.listdir(basepath)
    print(dirs)
    acclist = []
    losslist = []
    test_acclist = []
    dirs.sort()
    for i in range(len(dirs)):
        p = basepath + "/" + str(dirs[i]) + "/"
        if not merged:
            acclist.append(np.load(p + "accs.npy")[0:EPOCH_NUM])
            losslist.append(np.load(p + "losses.npy")[0:EPOCH_NUM])
            if cnn:
                test_acclist.append(np.load(p+"test_accs.npy")[0:EPOCH_NUM])
        else:
            acclist.append(np.load(p + "merged_accs.npy")[0:EPOCH_NUM])
            losslist.append(np.load(p + "merged_losses.npy")[0:EPOCH_NUM])
            if cnn:
                if not merged:
                    test_acclist.append(np.load(p+"merged_test_accs.npy")[0:EPOCH_NUM])
                else:
                    test_acclist.append(np.load(p+"test_accs.npy")[0:EPOCH_NUM])
    print("enumerating through results")
    for i,(acc, l) in enumerate(zip(acclist, losslist)):
        print("acc: ", acc.shape)
        print("l: ", l.shape)
        if cnn:
            print("test acc: ",test_acclist[i].shape)
    if cnn:
        return np.array(acclist), np.array(losslist), np.array(test_acclist)
    else:
        return np.array(acclist), np.array(losslist)

def get_lstm_results(basepath):
    return get_results(basepath, cnn=False)


def plot_results(pc_path, backprop_path,title):
    ### Plots initial results and accuracies ###
    acclist, losslist, test_acclist = get_results(pc_path)
    backprop_acclist, backprop_losslist, backprop_test_acclist = get_results(backprop_path)
    pc_list = [acclist, losslist, test_acclist]
    titles = ["accuracies", "losses", "test accuracies"]
    backprop_list = [backprop_acclist, backprop_losslist, backprop_test_acclist]
    print(acclist.shape)
    print(losslist.shape)
    print(test_acclist.shape)
    print(backprop_acclist.shape)
    print(backprop_losslist.shape)
    print(backprop_test_acclist.shape)
    xs = np.arange(0,EPOCH_NUM)
    for i,(pc, backprop) in enumerate(zip(pc_list, backprop_list)):
        mean_pc = np.mean(pc, axis=0)
        std_pc = np.std(pc,axis=0)
        mean_backprop = np.mean(backprop,axis=0)
        std_backprop = np.std(backprop,axis=0)
        fig,ax = plt.subplots(1,1)
        ax.fill_between(xs, mean_pc - std_pc, mean_pc+ std_pc, alpha=0.5,color='#228B22')
        plt.plot(mean_pc,label="Predictive coding",color='#228B22')
        ax.fill_between(xs, mean_backprop - std_backprop, mean_backprop+ std_backprop, alpha=0.5,color='#B22222')
        plt.plot(mean_backprop,label="Backprop",color='#B22222')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(title + " " + str(titles[i]),fontsize=18)
        ax.tick_params(axis='both',which='major',labelsize=12)
        ax.tick_params(axis='both',which='minor',labelsize=10)
        if titles[i] in ["accuracies", "test accuracies"]:
            plt.ylabel("Accuracy",fontsize=16)
        else:
            plt.ylabel("Loss")
        plt.xlabel("Iterations",fontsize=16)
        legend = plt.legend()
        legend.fontsize=14
        legend.style="oblique"
        frame  = legend.get_frame()
        frame.set_facecolor("1.0")
        frame.set_edgecolor("1.0")
        fig.tight_layout()
        fig.savefig("./figures/"+title +"_"+titles[i]+"_prelim_2.jpg")
        plt.show()

def plot_accs(pc_path, backprop_path,title):
    ### Plots training and test accuracies ###
    acclist, losslist, test_acclist = get_results(pc_path)
    backprop_acclist, backprop_losslist, backprop_test_acclist = get_results(backprop_path)
    fig, ax1 = plt.subplots(1,1)
    xs = np.arange(0,EPOCH_NUM)
    pc_acc = np.mean(acclist, axis=0)
    pc_std_acc = np.std(acclist,axis=0)
    bp_acc = np.mean(backprop_acclist,axis=0)
    bp_std_acc = np.std(backprop_acclist,axis=0)
    pc_test_acc = np.mean(test_acclist, axis=0)
    pc_test_std_acc = np.std(test_acclist,axis=0)
    bp_test_acc = np.mean(backprop_test_acclist,axis=0)
    bp_test_std_acc = np.std(backprop_test_acclist,axis=0)
    ax1.fill_between(xs, pc_acc - pc_std_acc, pc_acc+ pc_std_acc, alpha=0.3,color='#228B22')
    ax1.plot(pc_acc,label="Predictive coding train accuraacy",linestyle="--",color='#228B22')
    ax1.fill_between(xs, bp_acc - bp_std_acc, bp_acc+ bp_std_acc, alpha=0.3,color='#B22222')
    ax1.plot(bp_acc,label="Backprop train accuracy",linestyle="--",color='#B22222')

    ax1.fill_between(xs, pc_test_acc - pc_test_std_acc, pc_test_acc+ pc_test_std_acc, alpha=0.3,color='#228B22')
    ax1.plot(pc_test_acc,label="Predictive coding test accuracy",color='#228B22')
    ax1.fill_between(xs, bp_test_acc - bp_test_std_acc, bp_test_acc+ bp_test_std_acc, alpha=0.3,color='#B22222')
    ax1.plot(bp_test_acc,label="Backprop test accuracy",color='#B22222')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.xlabel("Epoch",fontsize=16,style='oblique')
    plt.ylabel("Accuracy",fontsize=16,style='oblique')
    plt.title(title + " CNN performance",fontsize=18)
    legend = plt.legend()
    legend.fontsize=10
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax1.tick_params(axis='both',which='major',labelsize=12)
    ax1.tick_params(axis='both',which='minor',labelsize=10)
    fig.savefig("./figures/"+title +"_CNN_prelim_2.jpg")
    plt.show()

def plot_weight_diffs(path):
    ### Plots the weight differences over training ###
    dirs = os.listdir(path)
    weights_list = []
    for i in range(len(dirs)):
        p = path + "/" + str(dirs[i]) + "/"
        weights_list.append(np.load(p+"weight_diffs.npy")[0:90])
    weights_list = np.array(weights_list)
    mean_weights_list = np.mean(weights_list,axis=0)
    std_weights_list = np.std(weights_list,axis=0)
    N,L = mean_weights_list.shape
    layer_names = ["Conv1","Conv2","Fc1","Fc2","Fc3"]
    for i in range(L):
        fig, ax = plt.subplots(1,1)
        xs = np.arange(0,90,1)
        plt.title("Layer : " + str(layer_names[i]) + " gradient divergence")
        ax.fill_between(xs,mean_weights_list[:,i]-std_weights_list[:,i],mean_weights_list[:,i] + std_weights_list[:,i],alpha=0.3)
        ax.plot(mean_weights_list[:,i])
        plt.ylabel("Mean distance from true gradient")
        plt.xlabel("Epoch")
        fig.savefig("./figures/cnn_weight_diff_"+str(i)+".jpg")
        plt.show()

def lstm_plot_results(pc_path, backprop_path,title,rnn=False):
    ### Plots the LSTM and RNN results ###
    acclist, losslist, = get_lstm_results(pc_path)
    backprop_acclist, backprop_losslist = get_lstm_results(backprop_path)

    pc_list = [acclist, losslist]
    titles = ["Accuracies", "Losses"]
    ylabels = ["Accuracy", "Loss"]
    backprop_list = [backprop_acclist, backprop_losslist]
    print(acclist.shape)
    print(losslist.shape)
    print(backprop_acclist.shape)
    print(backprop_losslist.shape)
    xs = np.arange(0,EPOCH_NUM)
    for i,(pc, backprop) in enumerate(zip(pc_list, backprop_list)):
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        plt.rcParams['axes.edgecolor'] = '#333F4B'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333F4B'
        plt.rcParams['ytick.color'] = '#333F4B'

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        cmap = plt.get_cmap("tab10")
        if rnn:
            # Convert from percentage to fraction accuracy for RNN since got this wrong in the RNN accuracy function
            pc = pc / 100
            backprop = backprop / 100

        mean_pc = np.mean(pc, axis=0)
        std_pc = np.std(pc,axis=0)
        mean_backprop = np.mean(backprop,axis=0)
        std_backprop = np.std(backprop,axis=0)
        fig,ax = plt.subplots(1,1)
        ax.fill_between(xs, mean_backprop - std_backprop, mean_backprop+ std_backprop, alpha=0.4,color='#228B22')
        plt.plot(mean_backprop,label="Backprop",color='#228B22',alpha=1)
        ax.fill_between(xs, mean_pc - std_pc, mean_pc+ std_pc, alpha=0.4,color='#B22222')
        plt.plot(mean_pc,label="Predictive coding",color='#B22222',alpha=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(title + " " + str(titles[i]),fontsize=16)
        plt.ylabel(ylabels[i],fontsize=14,style="oblique")
        plt.xlabel("Epochs",fontsize=14,style="oblique")
        ax.tick_params(axis='both',which='major',labelsize=12)
        ax.tick_params(axis='both',which='minor',labelsize=10)
        legend = plt.legend()
        legend.fontsize=14
        legend.style="oblique"
        frame  = legend.get_frame()
        frame.set_facecolor("1.0")
        frame.set_edgecolor("1.0")
        fig.tight_layout()
        fig.savefig("./figures/"+title +"_"+titles[i]+"_super_prelim_6.jpg")
        plt.show()
        #bib


def merge_results(basepath, merge_name,seeds=5,cnn=False):
    ### Performs the file merging from multiple sucessive runs ###
    dirs = os.listdir(basepath)
    mergelist = []
    for d in dirs:
        if merge_name in d:
            mergelist.append(str(d))
    mergelist.sort()
    base_dir = mergelist[0]
    for s in range(seeds):
        base_loss = np.load(basepath + base_dir +"/" + str(s) + "/losses.npy")
        base_acc =np.load(basepath +  base_dir  +"/" + str(s) + "/accs.npy")
        if cnn:
            base_test_accs = np.load(basepath + base_dir + "/" + str(s) + "/test_accs.npy")
        print("loss before: ", base_loss.shape)
        print("acc before: ", base_acc.shape)
        #print("test acc before: ", base_test_accs.shape)
        for i in range(1,len(mergelist)):
            mdir = mergelist[i]
            mloss = np.load(basepath +  "/" + mdir + "/" + str(s) + "/losses.npy")
            print("mloss: ", mloss.shape)
            macc =np.load(basepath  + mdir + "/" + str(s) + "/accs.npy")
            base_loss = np.concatenate((base_loss, mloss))
            base_acc = np.concatenate((base_acc, macc))
            if cnn:
                mtest_acc = np.load(basepath + mdir + "/" + str(s) + "/test_accs.npy")
                base_test_accs = np.concatenate((base_test_accs,mtest_acc))
        print("baseloss after: ", base_loss.shape)
        print("acc after: ", base_acc.shape)
        print("saving to: ",basepath+base_dir + "/"+str(s)+"/merged_losses.npy")
        np.save(basepath+base_dir + "/"+str(s)+"/merged_losses.npy", base_loss)
        np.save(basepath+base_dir + "/"+str(s)+"/merged_accs.npy", base_acc)
        if cnn:
            print("test acc after: ", base_test_accs.shape)
            np.save(basepath+base_dir + "/"+str(s)+"/merged_test_accs.npy", base_test_accs)


if __name__ == '__main__':
    #merge_results(pc_path,backprop_path,cnn=False)
    plot_results(pc_path,backprop_path,title)
    #plot_accs(pc_path,backprop_path,title)
    #plot_weight_diffs(pc_path)
    #lstm_plot_results(pc_path, backprop_path, title,rnn=True)
