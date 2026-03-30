#!/bin/bash 

# Slurm sbatch options 
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j
#SBATCH -c 8
#++++SBATCH --exclusive

# Loading the required module 
source /etc/profile 
module load anaconda/2022b

# load my light pyenv
MYPYENVROOT=/home/gridsan/mfeng1/my_python_user_bases
MYPYTHONENV=ccrl
MYPYTHONUSERBASE="${MYPYENVROOT}/$MYPYTHONENV"
export PATH=/home/gridsan/mfeng1/my_python_user_bases/$MYPYTHONENV/bin:$PATH
export PYTHONUSERBASE=$MYPYTHONUSERBASE

project_root=/home/gridsan/mfeng1/git_repos/cc-sorb
export PYTHONPATH=$project_root:$PYTHONPATH

env="FourRooms"
#env="CentralObstacle"

comment="cost_eval_max_dist_6"
experiment_dir="runs/results"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config="configs/config_SafePointEnv.yaml"
config="configs/config_PointEnv_Queue.yaml"
#config="configs/config_PointEnv_Queue_debug.yaml"
#config="configs/config_SafePointEnv_debug.yaml"

device="cpu"
#device="cuda:0"

cd "${project_root}"
python pud/algos/train_PointEnv.py \
    --cfg $config \
    --env $env \
    --logdir ${log_dir} \
    --device ${device}