# !/bin/sh

#debugger_port=5679

: '
screen command
screen_job sc3_staging_15 "conda_activate; conda activate habitat; bash launch_jobs/local_train_safe_habitat_lag.sh"
'


# sc02_staging_08
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sci_02_staging_08_linear_r1.txt"

ckpt="runs/results/habitat/job_local_sc0_staging_08/2024-09-10-02-59-43/ckpt/ckpt_0322500"
config="runs/results/habitat/job_local_sc0_staging_08/2024-09-10-02-59-43/bk/config.yaml"

ckpt="runs/results/habitat/job_local_sc0_staging_08/2024-09-10-03-00-39/ckpt/ckpt_0330000"
config="runs/results/habitat/job_local_sc0_staging_08/2024-09-10-03-00-39/bk/config.yaml"


illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sc0_staging_20_linear_r1.txt"
ckpt="runs/results/habitat/job_local_sc0_staging_20/2024-09-09-05-58-37/ckpt/ckpt_0480000"
config="runs/results/habitat/job_local_sc0_staging_20/2024-09-09-05-58-37/bk/config.yaml"

illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sc3_staging_05_linear_r1.txt"
ckpt="runs/results/habitat/job_local_sc3_staging_05/2024-09-09-05-57-11/ckpt/ckpt_0487500"
config="runs/results/habitat/job_local_sc3_staging_05/2024-09-09-05-57-11/bk/config.yaml"

illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sc3_staging_15_linear_r1.txt"
ckpt="runs/results/habitat/job_local_sc3_staging_15/2024-09-09-05-04-59/ckpt/ckpt_0562500"
config="runs/results/habitat/job_local_sc3_staging_15/2024-09-09-05-04-59/bk/config.yaml"

#illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sc3_staging_11_linear_r1.txt"
#ckpt="runs/hatbitat/job_26928411_visual_w_cost/2024-08-31-03-23-27/ckpt/ckpt_0720000"
#config="runs/hatbitat/job_26928411_visual_w_cost/2024-08-31-03-23-27/bk/config.yaml"


lambda_lr=0.035
collect_steps=20
eval_interval=2500  # 5000 | 10
num_iterations=600000
cost_limit=10

sampler_cost_bounds="0-40"
sampler_dist_bounds="0-5"
sampler_K=10
sampler_std_ub=1

device="cuda:1" # must use GPU cluster

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_safe_habitat_lag.py \
        --cfg $config \
        --ckpt $ckpt \
        --collect_steps $collect_steps \
        --eval_interval 20 \
        --cost_limit $cost_limit \
        --lambda_lr $lambda_lr \
        --num_iterations $num_iterations \
        --device ${device} \
        --illustration_pb_file ${illustration_pbs} \
        --sampler_cost_bounds $sampler_cost_bounds \
        --sampler_dist_bounds $sampler_dist_bounds \
        --sampler_K $sampler_K \
        --sampler_std_ub $sampler_std_ub \
        --visual \
        --pbar
else
    echo "[INFO] running in normal mode"
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
            --use_disk \
            --visual \
            --pbar
fi
