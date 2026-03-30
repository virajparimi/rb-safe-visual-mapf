#!/bin/bash

#SBATCH -c 20
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -o /home/gridsan/vparimi/cc-sorb-rev/runs/job.log-%j

# Loading the required module
source /etc/profile
module unload anaconda
module load anaconda/2023a-pytorch
module load cuda/11.8
module load nccl/2.18.1-cuda11.8

source activate hb

project_root=/home/gridsan/vparimi/cc-sorb-rev
export PYTHONPATH=$project_root:$PYTHONPATH

baseline=$1
baseline_options=("iddpg" "ippo" "isac" "maddpg" "mappo" "masac")
if [[ " ${baseline_options[@]} " =~ " ${baseline} " ]]; then
    # Call the corresponding Python file based on the argument
    case $baseline in
        "iddpg")
            python -u pud/algos/baselines/iddpg_PointEnv.py
            ;;
        "ippo")
            python -u pud/algos/baselines/ippo_PointEnv.py
            ;;
        "isac")
            python -u pud/algos/baselines/isac_PointEnv.py
            ;;
        "maddpg")
            python -u pud/algos/baselines/maddpg_PointEnv.py
            ;;
        "mappo")
            python -u pud/algos/baselines/mappo_PointEnv.py
            ;;
        "masac")
            python -u pud/algos/baselines/masac_PointEnv.py
            ;;
    esac
else
    # Print an error message if the argument is not valid
    echo $baseline
    echo "Error: Invalid argument. Please choose from: ${baseline_options[*]}"
fi
