#!/bin/bash
#SBATCH --exclusive --gres=gpu:volta:1
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j

source /etc/profile 
module load anaconda/2023a-pytorch
module load cuda/11.8
module load nccl/2.18.1-cuda11.8


# Slurm sbatch options, the last one is loaded --gres=gpu:volta:1
#SBATCH --gres=gpu:volta:1

#+++++SBATCH --exclusive
#++++ SBATCH -c 8
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs_debug/logs/job.log-%j
#++++ SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j
# Loading the required module 


source activate /home/gridsan/mfeng1/.conda/envs/hb

# load my light pyenv
MYPYENVROOT=/home/gridsan/mfeng1/my_python_user_bases
MYPYTHONENV=hb
MYPYTHONUSERBASE="${MYPYENVROOT}/$MYPYTHONENV"
export PATH=/home/gridsan/mfeng1/my_python_user_bases/$MYPYTHONENV/bin:$PATH
export PYTHONUSERBASE=$MYPYTHONUSERBASE
# add project to python path
project_root=/home/gridsan/mfeng1/git_repos/cc-sorb
export PYTHONPATH=$project_root:$PYTHONPATH
## -----------------------------------------------------------------------------
env=hatbitat
comment="venv=1"
#SLURM_JOB_ID=local_vec
experiment_dir="runs"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

eval_interval=2500  # 5000 | 10
#eval_interval=10
encoder=VisualEncoder

#resume=runs_debug/hatbitat/job_local_vec_fix_vec_step/2024-07-06-01-06-09/ckpt/ckpt_0460000

#config=configs/config_SafeHabitatEnv.yaml
#config=configs/config_SafeHabitatEnv_Queue_debug.yaml
#config=configs/config_HabitatEnv.yaml
config=configs/config_HabitatReplicaCAD.yaml
device="cuda:0" # must use GPU cluster

cd "${project_root}"

scene="sc0_staging_20"
scene="sc2_staging_08"
scene="sc3_staging_05"
scene="sc3_staging_11"
#scene="sc3_staging_15"
cost_name="linear"
cost_radius=1
num_envs=1
embedding_size=256

python pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py \
    --cfg $config \
    --scene $scene \
    --actor_lr 1e-5 \
    --critic_lr 1e-4 \
    --replay_buffer_size 100000 \
    --eval_interval $eval_interval \
    --encoder $encoder \
    --cost_name $cost_name \
    --cost_radius $cost_radius \
    --logdir ${log_dir} \
    --device ${device} \
    --visual \
    --num_envs ${num_envs} \
    --embedding_size $embedding_size