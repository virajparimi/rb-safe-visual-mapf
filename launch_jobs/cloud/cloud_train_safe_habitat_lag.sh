#!/bin/bash
#SBATCH --exclusive --gres=gpu:volta:1
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j

source /etc/profile 
module load anaconda/2023a-pytorch
module load cuda/11.8
module load nccl/2.18.1-cuda11.8


source activate /home/gridsan/mfeng1/.conda/envs/hb

# load my light pyenv
MYPYENVROOT=/home/gridsan/mfeng1/my_python_user_bases
MYPYTHONENV=hb
MYPYTHONUSERBASE="${MYPYENVROOT}/$MYPYTHONENV"
export PATH=/home/gridsan/mfeng1/my_python_user_bases/$MYPYTHONENV/bin:$PATH
export PYTHONUSERBASE=$MYPYTHONUSERBASE
# add project to python path
project_root=/home/gridsan/mfeng1/git_repos/cc-sorb
cd "${project_root}"
export PYTHONPATH=$project_root:$PYTHONPATH
## -----------------------------------------------------------------------------
#env=hatbitat
#comment="visual_cost_correct_flag"
##SLURM_JOB_ID=local_vec
#experiment_dir="runs"
#log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

#echo "project root directory: ${project_root}"
#echo "experiment directory: ${log_dir}"

lambda_lr=1
collect_steps=20
eval_interval=5000  # 5000 | 10
num_iterations=600000
cost_limit=10

sampler_cost_bounds="0-10"
sampler_dist_bounds="0-5"
sampler_K=10
sampler_std_ub=1

device="cuda:0" # must use GPU cluster
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sci_02_staging_08_linear_r1.txt"
ckpt="runs/hatbitat/job_26896380_visual_cost_correct_flag/2024-08-28-03-58-20/ckpt/ckpt_1582500"
config="runs/hatbitat/job_26896380_visual_cost_correct_flag/2024-08-28-03-58-20/bk/config.yaml"

illustration_pb_file=pud/envs/safe_habitatenv/illustration_set/sc3_staging_11_linear_r1.txt
config=runs/hatbitat/job_26928411_visual_w_cost/2024-08-31-03-23-27/bk/config.yaml
ckpt=runs/hatbitat/job_26928411_visual_w_cost/2024-08-31-03-23-27/ckpt/ckpt_0720000

python pud/algos/train_safe_habitat_lag.py \
        --cfg $config \
        --ckpt $ckpt \
        --collect_steps $collect_steps \
        --eval_interval $eval_interval \
        --cost_limit $cost_limit \
        --lambda_lr $lambda_lr \
        --num_iterations $num_iterations \
        --device ${device} \
        --illustration_pb_file ${illustration_pbs} \
        --sampler_cost_bounds $sampler_cost_bounds \
        --sampler_dist_bounds $sampler_dist_bounds \
        --sampler_K $sampler_K \
        --sampler_std_ub $sampler_std_ub \
        --visual