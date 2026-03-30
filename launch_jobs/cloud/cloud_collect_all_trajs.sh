#!/bin/bash
#SBATCH --time=4-00:00:00
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

env=$1

if [[ $env = "sc0_staging_20" ]]; then
    unconstrained_ckpt=models/SC0_Staging_20/ckpt/ckpt_0482500
    config=models/SC0_Staging_20/lag/2024-09-11-19-43-42/bk/config.yaml
    constrained_ckpt=models/SC0_Staging_20/lag/2024-09-11-19-43-42/ckpt/ckpt_0250000
elif [[ $env = "sc3_staging_05" ]]; then
    unconstrained_ckpt=models/SC3_Staging_05/ckpt/ckpt_0490000
    config=models/SC3_Staging_05/lag/2024-09-11-19-44-18/bk/config.yaml
    constrained_ckpt=models/SC3_Staging_05/lag/2024-09-11-19-44-18/ckpt/ckpt_0207500
elif [[ $env = "sc3_staging_11" ]]; then
    unconstrained_ckpt=models/SC3_Staging_11/ckpt/ckpt_0722500
    config=models/SC3_Staging_11/lag/2024-09-11-15-53-23/bk/config.yaml
    constrained_ckpt=models/SC3_Staging_11/lag/2024-09-11-15-53-23/ckpt/ckpt_0460000
elif [[ $env = "sc3_staging_15" ]]; then
    unconstrained_ckpt=models/SC3_Staging_15/ckpt/ckpt_0565000
    config=models/SC3_Staging_15/lag/2024-09-11-19-44-43/bk/config.yaml
    constrained_ckpt=models/SC3_Staging_15/lag/2024-09-11-19-44-43/ckpt/ckpt_0247500
else
    echo "Error: Invalid env provided. Please choose from: sc2_staging_20, sc3_staging_05, sc3_staging_11, sc3_staging_15"
    exit 1
fi

## -----------------------------------------------------------------------------
# pud/plots/collect_all_trajs.sh $env $config $unconstrained_ckpt $constrained_ckpt true 
pud/plots/collect_all_trajs.sh $env $config $unconstrained_ckpt $constrained_ckpt false 
