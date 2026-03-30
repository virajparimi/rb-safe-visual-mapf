#!/bin/bash

while true; do
    # python pud/plots/collect_trajectory_records.py --config_file runs_debug/UVFDDPG/CenterDot/job_530956.centerdot/2024-07-30-16-52-28/lag/2024-07-30-21-31-48/bk/bk_config.yaml --constrained_ckpt_file runs_debug/UVFDDPG/CenterDot/job_530956.centerdot/2024-07-30-16-52-28/lag/2024-07-30-21-31-48/ckpt/ckpt_0320000 --unconstrained_ckpt_file runs_debug/UVFDDPG/CenterDot/job_530956.centerdot/2024-07-30-16-52-28/ckpt/ckpt_0300000 --load_problem_set --problem_set_file pud/plots/illustration.npz --method_type constrained_search --num_agents 4
    python pud/plots/collect_trajectory_records.py --config_file models/SC2_Staging_08/lag/2024-08-30-06-17-29/bk/config.yaml --unconstrained_ckpt_file models/SC2_Staging_08/ckpt/ckpt_1587500 --constrained_ckpt_file models/SC2_Staging_08/lag/2024-08-30-06-17-29/ckpt/ckpt_0217500 --load_problem_set --problem_set_file pud/plots/data/sc2_staging_08_cost_mean_dist_4/illustration.npz --num_samples 50 --visual --method_type constrained_search --num_agents 5
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script completed successfully."
        break
    fi
    echo "Script crashed with exit code $EXIT_CODE. Restarting..." >&2
sleep 1
done
