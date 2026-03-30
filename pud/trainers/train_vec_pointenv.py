import sys
import yaml
import torch
import random
import numpy as np
from typing import List
from pathlib import Path
from dotmap import DotMap
from termcolor import cprint
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.ddpg import UVFDDPG
from pud.utils import set_global_seed
from pud.buffers.buffer import ReplayBuffer
from pud.algos.policies import GaussianPolicy
from pud.envs.simple_navigation_env import env_load_fn
from pud.runners.runner_vec import train_eval, eval_pointenv_dists


def setup_logger(
    root_dir: str, subdir_names: List[str], tag_time: bool = False, verbose=True
):
    log_dir = Path(root_dir)
    if tag_time:
        from datetime import datetime

        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = log_dir.joinpath(date_time)

    log_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        cprint("[Logger] root directory: {}".format(log_dir.as_posix()), "green")
    logger = {"log_dir": log_dir}
    for name in subdir_names:
        subdir = log_dir.joinpath(name)
        subdir.mkdir(parents=True, exist_ok=True)
        logger[name] = subdir

    return logger


num_envs = 4

cfg = {}
with open(sys.argv[-1], "r") as f:
    cfg = yaml.safe_load(f)
# for dot completion
cfg = DotMap(cfg)
cfg.num_envs = num_envs
cfg.runner.num_iterations = 10000
cfg.pprint()
set_global_seed(cfg.seed)

# Custom Logging
logger = setup_logger(
    root_dir=cfg.ckpt_dir,
    subdir_names=[
        "ckpt",
        "tfevent",
        "bk",
    ],  # "imgs"
    tag_time=True,
)
with open(logger["bk"].joinpath("config.yaml"), "w") as f:
    yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
logger["tb"] = SummaryWriter(log_dir=logger["tfevent"].as_posix())  # type: ignore


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

envs = [
    env_load_fn(
        cfg.env.env_name,
        cfg.env.max_episode_steps,
        resize_factor=cfg.env.resize_factor,
        terminate_on_timeout=False,
        thin=cfg.env.thin,
    )
    for _ in range(num_envs)
]

env = envs[0]  # To help initialize other modules

eval_env = env_load_fn(
    cfg.env.env_name,
    cfg.env.max_episode_steps,
    resize_factor=cfg.env.resize_factor,
    terminate_on_timeout=True,
    thin=cfg.env.thin,
)

obs_dim = env.observation_space["observation"].shape[0]  # type: ignore
goal_dim = obs_dim
state_dim = obs_dim + goal_dim

assert env.action_space.shape is not None
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])  # type: ignore
print(
    f"obs dim: {obs_dim}, goal dim: {goal_dim}, state dim: {state_dim}, "
    f"action dim: {action_dim}, max action: {max_action}"
)

agent = UVFDDPG(
    state_dim,  # Concatenating obs and goal
    action_dim,
    max_action,
    **cfg.agent,
)
print(agent)

policy = GaussianPolicy(agent)
logger["eval_distances"] = [2, 5, 10]  # type: ignore
cfg.replay_buffer.max_size = cfg.replay_buffer.max_size * num_envs
replay_buffer = ReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)

train_eval(
    policy,
    agent,
    replay_buffer,
    env=envs,
    logger=logger,
    eval_env=eval_env,
    eval_func=eval_pointenv_dists,
    **cfg.runner,
)
torch.save(agent.state_dict(), logger["ckpt"].joinpath("agent.pth"))
