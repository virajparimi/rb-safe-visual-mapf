#!/bin/bash

#SBATCH -c 20
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH -o /home/gridsan/vparimi/cc-sorb-rev/runs/job.log-%j

# Loading the required module
source /etc/profile
module unload anaconda
module load anaconda/2023a-pytorch

source activate habitat

project_root=/home/gridsan/vparimi/cc-sorb-rev
export PYTHONPATH=$project_root:$PYTHONPATH

env=$1
SLURM_JOB_ID="_${STY}"
experiment_dir="runs_debug/results"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}

echo "Project Root Directory: ${project_root}"
echo "Experiment Directory: ${log_dir}"

device="cpu"
config="configs/config_PointEnv_Queue_${env}.yaml"

cd "${project_root}"

resize_factor=5
actor_lr=0.0001
critic_lr=0.0003
action_noise=0.0

cost_radius=3.0
cost_name="linear"

illustration_pb_file="pud/envs/safe_pointenv/illustration_set/FourRoomsModified_resize_5_linear_r3_4pts.txt"

echo "[INFO] Running in normal mode"
python pud/algos/train_PointEnv.py \
    --cfg $config \
    --env $env \
    --action_noise $action_noise \
    --actor_lr $actor_lr \
    --critic_lr $critic_lr \
    --resize_factor $resize_factor \
    --cost_name $cost_name \
    --cost_radius $cost_radius \
    --logdir ${log_dir} \
    --device ${device} \
    --illustration_pb_file $illustration_pb_file \
    --visual \
    --pbar
