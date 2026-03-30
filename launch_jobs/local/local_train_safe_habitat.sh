# !/bin/sh

env=hatbitat
comment="debug_safe"
SLURM_JOB_ID=local_debug
experiment_dir="runs_debug/results"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

cd "${project_root}"

#debugger_port=5679

#scene="sc0_staging_20"
scene="sc2_staging_08"
#scene="sc3_staging_05"
#scene="sc3_staging_11"
#scene="sc3_staging_15"

eval_interval=2500  # 5000 | 10
encoder=VisualEncoder
cost_name=linear
cost_radius=1
cost_limit=10
num_envs=1
embedding_size=256
config=configs/config_SafeHabitatReplicaCAD.yaml
device="cuda:0" # must use GPU cluster
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sci_02_staging_08_linear_r1.txt"
# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_safe_habitat.py \
        --cfg $config \
        --scene $scene \
        --actor_lr 1e-5 \
        --critic_lr 1e-4 \
        --replay_buffer_size 100000 \
        --eval_interval 20 \
        --encoder $encoder \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --cost_limit $cost_limit \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --illustration_pb_file ${illustration_pbs} \
        --embedding_size $embedding_size \
        --pbar
else
    echo "[INFO] running in normal mode"
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
        --embedding_size $embedding_size \
        --pbar
fi