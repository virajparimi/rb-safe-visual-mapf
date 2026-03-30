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
export PYTHONPATH=$project_root:$PYTHONPATH
## -----------------------------------------------------------------------------
env=hatbitat
comment="visual"
#SLURM_JOB_ID=local_vec
experiment_dir="runs"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"


cd "${project_root}"

scene="sc0_staging_20"
scene="sc2_staging_08"
scene="sc3_staging_05"
scene="sc3_staging_11"
#scene="sc3_staging_15"

eval_interval=2500  # 5000 | 10
encoder=VisualEncoder
cost_name=linear
cost_radius=1
cost_limit=10
embedding_size=256
config=configs/config_SafeHabitatReplicaCAD.yaml
device="cuda:0" # must use GPU cluster
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sci_02_staging_08_linear_r1.txt"

python pud/algos/train_safe_habitat.py \
    --cfg $config \
    --scene $scene \
    --actor_lr 1e-5 \
    --critic_lr 1e-4 \
    --replay_buffer_size 100000 \
    --eval_interval $eval_interval \
    --encoder $encoder \
    --cost_name $cost_name \
    --cost_radius $cost_radius \
    --cost_limit $cost_limit \
    --logdir ${log_dir} \
    --device ${device} \
    --visual \
    --illustration_pb_file ${illustration_pbs} \
    --embedding_size $embedding_size