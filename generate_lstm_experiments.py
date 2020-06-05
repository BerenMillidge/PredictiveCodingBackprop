import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
network_type = str(sys.argv[4])
exp_name = str(sys.argv[5])
old_savename = "None"
base_call = "python lstm.py" + " --network_type " + network_type
output_file = open(generated_name, "w")
seeds = 5
learning_rates = [0.0005,0.0001,0.001,0.005,0.00005]
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --old_savedir " + str(old_savepath)
    print(final_call)
    print(final_call, file=output_file)
