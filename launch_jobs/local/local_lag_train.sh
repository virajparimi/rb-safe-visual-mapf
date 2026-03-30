# !/bin/sh

device="cuda:0"
#device="cuda:1"

cd "${project_root}"

#ckpt="runs_debug/results/CenterDot/job_530956.centerdot/2024-07-30-16-52-28/ckpt/ckpt_0293000"
ckpt="runs_debug/results/FourRoomsModified/job_/2024-08-29-17-00-49/ckpt/ckpt_0300000"
#config="runs_debug/results/CenterDot/job_530956.centerdot/2024-07-30-16-52-28/bk/bk_config.yaml"
config=runs_debug/results/FourRoomsModified/job_/2024-08-29-17-00-49/bk/bk_config.yaml
lambda_lr=10
#illustration_pb_file="pud/envs/safe_pointenv/illustration_set/CenterDot_resize_3_linear_r3.txt"
illustration_pb_file="pud/envs/safe_pointenv/illustration_set/FourRoomsModified_resize_5_linear_r3_4pts.txt"
cost_limit=10.0

num_iterations=600000
collect_steps=20

#debugger_port=5678

if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_lag_policy.py \
            --cfg $config \
            --collect_steps $collect_steps \
            --ckpt $ckpt \
            --device ${device} \
            --cost_limit $cost_limit \
            --illustration_pb_file $illustration_pb_file \
            --lambda_lr $lambda_lr \
            --num_iterations $num_iterations \
            --eval_interval 10 \
            --initial_collect_steps 10 \
            --visual \
            --pbar \
            --verbose
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_lag_policy.py \
        --cfg $config \
        --collect_steps $collect_steps \
        --ckpt $ckpt \
        --device ${device} \
        --cost_limit $cost_limit \
        --illustration_pb_file $illustration_pb_file \
        --lambda_lr $lambda_lr \
        --num_iterations $num_iterations \
        --visual \
        --pbar
fi
