# !/bin/sh

#env="FourRooms"
env="CentralObstacle"
env="LQuarter"
env="Line"
env="CenterDot"
env="FourRoomsModified"

#comment="cost_limit=10"
SLURM_JOB_ID="_${STY}"
experiment_dir="runs_debug/results"
log_dir=${experiment_dir}/${env}/job${SLURM_JOB_ID}${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config="configs/config_SafePointEnv.yaml"
config="configs/config_PointEnv_Queue.yaml"
config="configs/config_PointEnv_Queue_debug.yaml"
config="configs/config_PointEnv_Queue_LQuarter.yaml"
config="configs/config_PointEnv_Queue_Line.yaml"
config="configs/config_PointEnv_Queue_CenterDot.yaml"
config="configs/config_PointEnv_Queue_FourRoomsModified.yaml"
#config="configs/config_PointEnv_Queue_LQuarter_debug.yaml"
#config="configs/config_SafePointEnv_debug.yaml"

#device="cpu"
device="cuda:0"

cd "${project_root}"

#debugger_port=5678

#cost_max=
#cost_N=

#cost_name="linear"
#cost_radius=2
#illustration_pb_file="pud/envs/safe_pointenv/illustration_set/Line_resize_1_linear_r2.txt"

#cost_name="linear"
#cost_radius=6
#illustration_pb_file="pud/envs/safe_pointenv/illustration_set/Line_resize_3_linear_r6.txt"
#action_noise=0.0
#resize_factor=3

#cost_name="linear"
#cost_radius=5.0
#illustration_pb_file="pud/envs/safe_pointenv/illustration_set/Line_resize_2_linear_r5.txt"
#action_noise=0.0
#resize_factor=2

cost_name="linear"
cost_radius=8.0
illustration_pb_file="pud/envs/safe_pointenv/illustration_set/Line_resize_4_linear_r8.txt"
action_noise=0.0
resize_factor=4
actor_lr=0.0001
critic_lr=0.0003


cost_name="linear"
cost_radius=3.0
illustration_pb_file="pud/envs/safe_pointenv/illustration_set/CenterDot_resize_3_linear_r3.txt"
action_noise=0.0
resize_factor=3
actor_lr=0.0001
critic_lr=0.0003

resize_factor=5
actor_lr=0.0001
critic_lr=0.0003
action_noise=0.0
cost_radius=3.0
cost_name="linear"
illustration_pb_file="pud/envs/safe_pointenv/illustration_set/FourRoomsModified_resize_5_linear_r3_4pts.txt"


# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_PointEnv.py \
        --cfg $config \
        --env $env \
        --action_noise $action_noise \
        --resize_factor $resize_factor \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --illustration_pb_file $illustration_pb_file \
        --visual \
        --pbar
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_PointEnv.py \
        --cfg $config \
        --env $env \
        --action_noise $action_noise \
        --actor_lr $actor_lr \
        --critic_lr $critic_lr \
        --resize_factor $resize_factor \
        --cost_name $cost_name \
        --cost_radius $cost_radius \
        --logdir ${log_dir} \
        --device ${device} \
        --illustration_pb_file $illustration_pb_file \
        --visual \
        --pbar
fi
