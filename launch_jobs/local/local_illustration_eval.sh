# !/bin/sh

cfg_path="runs_debug/results/Line/job_462169.max_cost_40_lr1e-4/2024-07-24-22-04-34/bk/bk_config.yaml"
ckpt_path="runs_debug/results/Line/job_462169.max_cost_40_lr1e-4/2024-07-24-22-04-34/ckpt/ckpt_0300000"
figdir="temp"
illustration_pb_file=pud/envs/safe_pointenv/illustration_set/Line_resize_4_linear_r8.txt

python pud/algos/vis_policy_trajs.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir \
    --illustration_pb_file $illustration_pb_file