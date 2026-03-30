#!/bin/bash

DO_SAMPLE=$1
num_samples=50
agents=(20)
problem_types=("hard" "medium" "easy")
method_types=(
    "collect_bounds_data"
    "constrained_risk_search" 
    "lagrangian_search" 
    "biobjective_search"
    "surplus_driven_uniform_search" # equiris
    "tatonnement_driven_uniform_search"  # walris
)

collect_trajectories() {
    while true; do
        taskset -c 0 python -u pud/plots/collect_safe_trajectory_records.py              \
            --config_file "$config_file"                                    \
            --unconstrained_ckpt_file "$unconstrained_ckpt_file"            \
            --constrained_ckpt_file "$constrained_ckpt_file"                \
            --load_problem_set --problem_set_file "$problem_set_file"       \
            --num_samples "$num_samples"                                    \
            --method_type "$method_type"                                    \
            --num_agents "$num_agent"                                       \
            --traj_difficulty "$problem_type"                               \
            "$visual"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo " OK: ${method_type} (${num_agent} agents, ${problem_type}) completed successfully."
            break
        fi
        echo "Script crashed with exit code $EXIT_CODE. Restarting ${method_type} (${num_agent} agents, ${problem_type})..." >&2
    sleep 1
    done
}

# envs=("sc0_staging_20" "sc2_staging_08" "sc3_staging_05" "sc3_staging_11" "sc3_staging_15" "centerdot") 
envs=("centerdot")

for env in "${envs[@]}"; do
    echo
    echo "=============================================="
    echo "  👉  Processing environment: $env"
    echo "=============================================="

    case "$env" in
        "sc0_staging_20")
            config_file=models/SC0_Staging_20/lag/2024-09-11-19-43-42/bk/config.yaml
            unconstrained_ckpt_file=models/SC0_Staging_20/ckpt/ckpt_0482500
            constrained_ckpt_file=models/SC0_Staging_20/lag/2024-09-11-19-43-42/ckpt/ckpt_0250000
            ;;
        "sc3_staging_05")
            config_file=models/SC3_Staging_05/lag/2024-09-11-19-44-18/bk/config.yaml
            unconstrained_ckpt_file=models/SC3_Staging_05/ckpt/ckpt_0490000
            constrained_ckpt_file=models/SC3_Staging_05/lag/2024-09-11-19-44-18/ckpt/ckpt_0207500
            ;;
        "sc3_staging_11")
            config_file=models/SC3_Staging_11/lag/2024-09-11-15-53-23/bk/config.yaml
            unconstrained_ckpt_file=models/SC3_Staging_11/ckpt/ckpt_0722500
            constrained_ckpt_file=models/SC3_Staging_11/lag/2024-09-11-15-53-23/ckpt/ckpt_0460000
            ;;
        "sc3_staging_15")
            config_file=models/SC3_Staging_15/lag/2024-09-11-19-44-43/bk/config.yaml
            unconstrained_ckpt_file=models/SC3_Staging_15/ckpt/ckpt_0565000
            constrained_ckpt_file=models/SC3_Staging_15/lag/2024-09-11-19-44-43/ckpt/ckpt_0247500
            ;;
        "sc2_staging_08")
            config_file=models/SC2_Staging_08/lag/2024-09-11-19-42-08/bk/config.yaml
            unconstrained_ckpt_file=models/SC2_Staging_08/ckpt/ckpt_0325000
            constrained_ckpt_file=models/SC2_Staging_08/lag/2024-09-11-19-42-08/ckpt/ckpt_0255000
            ;;
        "centerdot")
            config_file=models/CenterDot/lag/2024-07-30-21-31-48/bk/bk_config.yaml
            unconstrained_ckpt_file=models/CenterDot/ckpt/ckpt_0300000
            constrained_ckpt_file=models/CenterDot/lag/2024-07-30-21-31-48/ckpt/ckpt_0600000
            agents=(20)
            ;;
        *)
            echo "Unknown environment '$env' – skipping."
            continue
            ;;
    esac


    if [[ $env == *"staging"* ]]; then
        visual="--visual"
    else
        visual=""
    fi

    if [[ $DO_SAMPLE = true ]]; then
        for problem_type in "${problem_types[@]}"; do
            printf "%*s\n" 100 | tr ' ' '*'
            echo "Sampling problems for ${env} on ${problem_type} problems"
            taskset -c 0 python -u pud/plots/collect_safe_trajectory_records.py              \
            --config_file "$config_file"                                        \
            --unconstrained_ckpt_file "$unconstrained_ckpt_file"                \
            --constrained_ckpt_file "$constrained_ckpt_file"                    \
            --collect_trajs --traj_difficulty "$problem_type"                   \
            --num_samples "$num_samples" --num_agents 25 "$visual"
        done
    fi

    # Collect the trajectories
    for problem_type in "${problem_types[@]}"; do
        problem_set_file=pud/plots/data/${env}/${problem_type}.npz
        for num_agent in "${agents[@]}"; do
            printf "%*s\n" 100 | tr ' ' '-'
            echo "Collecting trajectories for ${env} with ${num_agent} agents on ${problem_type} problems"
            printf "%*s\n" 100 | tr ' ' '-'
            for method_type in "${method_types[@]}"; do
                printf "%*s\n" 50 | tr ' ' '*'
                echo "Method type: ${method_type}"
                printf "%*s\n" 50 | tr ' ' '*'
                collect_trajectories
            done
        done
    done

    echo "Finished processing environment: $env"
    echo "=============================================="
done

echo "All environments processed."

