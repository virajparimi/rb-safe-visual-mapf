# !/bin/sh

env=hatbitat
comment="fix_vec_step"
SLURM_JOB_ID=local_vec
experiment_dir="runs_debug"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config=configs/config_SafeHabitatEnv.yaml
#config=configs/config_SafeHabitatEnv_Queue_debug.yaml
#config=configs/config_HabitatEnv.yaml
config=configs/config_HabitatReplicaCAD.yaml
#device="cpu"
device="cuda:0"

cd "${project_root}"

debugger_port=5678

cost_name="linear"
cost_radius=2
num_envs=1
embedding_size=256
encoder=VisualEncoder # VisualRGBEncoder | VisualEncoder
eval_interval=5000  # 5000 | 10
#resume=runs_debug/hatbitat/job_26450336_small_lr_encoder_v1/2024-07-04-17-16-21/ckpt/ckpt_0972000

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py \
        --cfg $config \
        --scene "sc2_staging_08" \
        --actor_lr 1e-6 \
        --critic_lr 1e-5 \
        --replay_buffer_size 10000 \
        --eval_interval 10 \
        --encoder $encoder \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --num_envs ${num_envs} \
        --embedding_size $embedding_size \
        --pbar
else
    echo "[INFO] running in normal mode"
    python pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py \
        --cfg $config \
        --scene "sc2_staging_08" \
        --actor_lr 1e-6 \
        --critic_lr 1e-5 \
        --replay_buffer_size 10000 \
        --eval_interval $eval_interval \
        --encoder $encoder \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --num_envs ${num_envs} \
        --embedding_size $embedding_size \
        --pbar
fi
