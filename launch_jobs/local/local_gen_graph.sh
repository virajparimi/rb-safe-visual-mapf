# !/bin/sh

#ckpt_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0260000"
#cfg_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"
#figdir="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/visuals"

cfg_path="runs_debug/results/Line/job_412488.r4/2024-07-24-15-39-34/bk/bk_config.yaml"
ckpt_path="runs_debug/results/Line/job_412488.r4/2024-07-24-15-39-34/ckpt/ckpt_0196000"
figdir="runs_debug/results/Line/job_412488.r4/2024-07-24-15-39-34/"
illustration_pb_file=pud/envs/safe_pointenv/illustration_set/Line_resize_4_linear_r8.txt

python pud/algos/gen_graph.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir \
    --illustration_pb_file $illustration_pb_file