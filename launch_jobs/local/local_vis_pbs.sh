# !/bin/sh

ckpt_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0260000"
cfg_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"
figdir="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/visuals"

K=100
N=200

# for dist
metric="dist"
target=20.0
min_dist=0
max_dist=20

# for cost
#metric="cost"
#target=2.0
#min_dist=0
#max_dist=3

target_str="none"
if [[ -n ${target} ]]; then
    target_str=${target}
fi

figname="pb_${metric}=${target_str}_dist_${min_dist}-${max_dist}_K=${K}_N=${N}.jpg"

python pud/algos/unit_tests/vis_sampler.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir \
    --K $K \
    --metric $metric \
    --target $target \
    --N $N \
    --min_dist $min_dist \
    --max_dist $max_dist \
    --figname $figname