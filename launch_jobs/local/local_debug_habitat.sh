# !/bin/sh

env=hatbitat
comment="cost_limit=10"
SLURM_JOB_ID=local_debug
experiment_dir="runs_debug"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config=configs/config_SafeHabitatEnv.yaml
#config=configs/config_SafeHabitatEnv_Queue_debug.yaml
config=configs/config_HabitatEnv.yaml
#device="cpu"
device="cuda:1"

cd "${project_root}"

#debugger_port=5678

cost_name="linear"
cost_radius=10.0
scene=scene_datasets/habitat-test-scenes/skokloster-castle.glb
apsp_path=pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/envs/safe_habitatenv/unit_tests/train_uvfddpg_habitat.py \
        --scene $scene \
        --apsp_path $apsp_path \
        --cfg $config \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --pbar
else
    echo "[INFO] running in normal mode"
    python pud/envs/safe_habitatenv/unit_tests/train_uvfddpg_habitat.py \
        --scene $scene \
        --apsp_path $apsp_path \
        --cfg $config \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --pbar
fi