import yaml
import torch
import shutil
import argparse
from typing import List
from pathlib import Path
from dotmap import DotMap
from gym.spaces import Box
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import VectorGaussianPolicy
from pud.utils import set_env_seed, set_global_seed
from pud.algos.vision.vision_agent import VisionUVFDDPG
from pud.buffers.visual_buffer import VisualReplayBuffer
from pud.runners.runner_vec import train_eval, eval_pointenv_dists
from pud.envs.habitat_navigation_env import (
    GoalConditionedHabitatPointWrapper,
    habitat_env_load_fn,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)


def setup_logger(root_dir: str, subdir_names: List[str], tag_time: bool = False):
    log_dir = Path(root_dir)
    if tag_time:
        from datetime import datetime

        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = log_dir.joinpath(date_time)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = {"log_dir": log_dir}
    for name in subdir_names:
        subdir = log_dir.joinpath(name)
        subdir.mkdir(parents=True, exist_ok=True)
        logger[name] = subdir

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_HabitatReplicaCAD.yaml",
        help="Training configuration",
    )
    parser.add_argument("--cost_name", type=str, default="", help="override cost type")
    parser.add_argument(
        "--cost_radius", type=float, default=-1, help="override cost type"
    )
    parser.add_argument("--cost_max", type=float, default=-1, help="override if > 0")
    parser.add_argument("--cost_N", type=int, default=0, help="override if > 0")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="number of envs for batch inference"
    )
    parser.add_argument("--embedding_size", type=int, default=-1, help="")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="if non-empty, load the checkpoint into agent model",
    )
    parser.add_argument(
        "--i_start",
        type=int,
        default=1,
        help="override the start iteration index, for resume training",
    )
    parser.add_argument(
        "--collect_steps",
        type=int,
        default=-1,
        help="override the number of steps per agent update",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=-1, help="override num of iterations"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=-1, help="evaluation interval"
    )
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="override cost limit"
    )
    parser.add_argument("--scene", type=str, default="", help="override scene")
    parser.add_argument("--actor_lr", type=float, default=-1, help="")
    parser.add_argument("--critic_lr", type=float, default=-1, help="")
    parser.add_argument(
        "--lambda_lr", type=float, default=-1, help="override lagrange lr"
    )
    parser.add_argument(
        "--replay_buffer_size", type=int, default=-1, help="max replay buffer size"
    )
    parser.add_argument(
        "--apsp_path", type=str, default="", help="supply pre-computed apsp path"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="VisualRGBEncoder",
        help="VisualRGBEncoder | VisualEncoder",
    )
    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        help="problems that serve as illustration and evaluation set",
    )
    parser.add_argument("--resume", type=str, default="", help="resume training")
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--visual", action="store_true", help="generate and save visual trajs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)

    # Override cfs from terminal
    if len(args.logdir) > 0:
        cfg.ckpt_dir = args.logdir
    if len(args.cost_name) > 0:
        cfg.cost_function.name = args.cost_name
    if args.cost_radius > 0:
        cfg.cost_function.radius = args.cost_radius
    if args.cost_max > 0:
        cfg.agent.cost_max = args.cost_max
    if args.cost_N > 0:
        cfg.agent.cost_N = args.cost_N
    if args.lambda_lr > 0:
        cfg.agent.lambda_lr = args.lambda_lr
    if args.actor_lr > 0:
        cfg.agent.actor_lr = args.actor_lr
    if args.critic_lr > 0:
        cfg.agent.critic_lr = args.critic_lr
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    if args.eval_interval > 0:
        cfg.runner.eval_interval = args.eval_interval
    if args.replay_buffer_size > 0:
        cfg.replay_buffer.max_size = args.replay_buffer_size
    if len(args.apsp_path) > 0:
        cfg.env.apsp_path = args.apsp_path
    if len(args.scene) > 0:
        cfg.env.simulator_settings.scene = args.scene
    if args.embedding_size > 0:
        cfg.agent.embedding_size = args.embedding_size
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    cfg.pprint()

    # Custom Logging
    logger = setup_logger(
        root_dir=cfg.ckpt_dir,
        subdir_names=["ckpt", "tfevent", "bk", "imgs"],
        tag_time=True,
    )
    with open(logger["bk"].joinpath("config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    logger["tb"] = SummaryWriter(log_dir=logger["tfevent"].as_posix())  # type: ignore

    shutil.copy("launch_jobs/cloud_debug_vec_habitat.sh", logger["bk"].as_posix())
    shutil.copy(
        "pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py",
        logger["bk"].as_posix(),
    )
    shutil.copy("pud/vision_agent.py", logger["bk"].as_posix())
    shutil.copy("pud/runner_vec.py", logger["bk"].as_posix())
    shutil.copy("pud/visual_models.py", logger["bk"].as_posix())

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "GoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(GoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    envs = [
        habitat_env_load_fn(
            **cfg.env.toDict(),
            max_episode_steps=cfg.time_limit.max_episode_steps,
            gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),  # type: ignore
            wrapper_kwargs=gym_env_wrapper_kwargs,
            device=cfg.device,
            terminate_on_timeout=False,
        )
        for _ in range(args.num_envs)
    ]
    for i in range(args.num_envs):
        set_env_seed(envs[i], cfg.seed + i)
    env = envs[0]  # To help initialize other modules

    eval_env = habitat_env_load_fn(
        **cfg.env.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        device=cfg.device,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + args.num_envs)

    assert env.action_space.shape is not None
    cfg.agent["action_dim"] = env.action_space.shape[0]
    assert isinstance(env.action_space, Box)
    cfg.agent["max_action"] = float(env.action_space.high[0])

    agent = VisionUVFDDPG(
        width=cfg.env.simulator_settings.width,
        height=cfg.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder=args.encoder,
        device=cfg.device,
        **cfg.agent.toDict(),
    )

    agent.to(device=cfg.device)

    if len(args.resume) > 0:
        state_dict = torch.load(args.resume)
        agent.load_state_dict(state_dict)

    # Test collector
    replay_buffer = VisualReplayBuffer(
        obs_dim=(
            4,
            cfg.env.simulator_settings.width,
            cfg.env.simulator_settings.height,
            4,
        ),
        goal_dim=(
            4,
            cfg.env.simulator_settings.width,
            cfg.env.simulator_settings.height,
            4,
        ),
        action_dim=env.action_space.shape[0],
        max_size=cfg.replay_buffer.max_size,
    )

    policy = VectorGaussianPolicy(agent, noise_scale=0.2)

    train_eval(
        policy,
        agent,
        replay_buffer,
        env=envs,
        eval_env=eval_env,
        eval_func=eval_pointenv_dists,
        logger=logger,
        **cfg.runner,
    )
    torch.save(agent.state_dict(), logger["ckpt"].joinpath("agent.pth"))
