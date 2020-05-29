#!/bin/bash


# FMI about options, see https://slurm.schedmd.com/sbatch.html
# N.B. options supplied on the command line will overwrite these set here

# *** To set any of these options, remove the first comment hash '# ' ***
# i.e. `# # SBATCH ...` -> `#SBATCH ...`

#SBATCH --job-name=TESETJOBNAME

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:2

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=16000

#amount of tasks run in parallel on each node - this is just a test!
# #SBATCH --ntasks-per-node=1

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
#  #SBATCH --time=7-00:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
# #SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]

source ~/.bashrc
set -e
echo "Setting up log files"
USER=s1686853
SCRATCH_DISK=/disk/scratch
log_path=${SCRATCH_DISK}/${USER}/initial_bipedal_experiments
mkdir -p ${log_path}

echo "Initializing Conda Environment"
CONDA_NAME=env
conda activate ${CONDA_NAME}
#mkdir -p ${log_path}


echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/home/${USER}/PredictiveCodingBackpropStaging
mnist_path=${repo_home}/mnist_data
cifar_path=${repo_home}/cifar_data
cifar100_path=${repo_home}/cifar100_data
svhn_path=${repo_home}/svhn_data
names_path=${repo_home}/data
#src_path=${repo_home}/experiments/examples/mnist/data/input

# input data directory path on the scratch disk of the node
mnist_dest_path=${SCRATCH_DISK}/${USER}/mnist_data
mkdir -p ${mnist_dest_path}  # make it if required
cifar_dest_path=${SCRATCH_DISK}/${USER}/cifar_data
mkdir -p ${cifar_dest_path}  # make it if required
cifar100_dest_path=${SCRATCH_DISK}/${USER}/cifar100_data
mkdir -p ${cifar100_dest_path}  # make it if required
svhn_dest_path=${SCRATCH_DISK}/${USER}/svhn_data
mkdir -p ${svhn_dest_path}  # make it if required
names_dest_path=${SCRATCH_DISK}/${USER}/data
mkdir -p ${names_dest_path}

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

rsync --archive --update --compress --progress ${mnist_path}/ ${mnist_dest_path}
echo "Rsynced mnist"
rsync --archive --update --compress --progress ${cifar_path}/ ${cifar_dest_path}
echo "Rsynced cifar"
rsync --archive --update --compress --progress ${cifar100_path}/ ${cifar100_dest_path}
echo "Rsynced cifar100"
rsync --archive --update --compress --progress ${svhn_path}/ ${svhn_dest_path}
echo "Rsynced svhn"
rsync --archive --update --compress --progress ${names_path}/ ${names_dest_path}
echo "Rsynced names data"

#echo "Running experiment command"
#pip install git+https://github.com/Bmillidgework/exploration-baselines
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
#python main.py --env_name "SparseBipedalWalker" --logdir ${log_path} --action_noise "0.1" --plan_horizon "40" --action_repeat "2" --ensemble_size "15" --use_exploration "True" --use_reward "True" --expl_scale "0.1" --n_episodes "10"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"
#echo "Experiment Finished. Moving data back to DFS"
#echo "log path: ${log_path}"
#save_path=/home/${USER}/fe_mbrl/bipedal_walker_initial_tests/${SLURM_ARRAY_TASK_ID}
#mkdir -p ${save_path}
#echo "save_path: ${save_path}"
#rsync --archive --update --compress --progress ${log_path}/ ${save_path}

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
