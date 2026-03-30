import yaml
import torch
import shutil
import argparse
from typing import List
from pathlib import Path
from dotmap import DotMap
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import GaussianPolicy
from pud.utils import set_env_seed, set_global_seed
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.buffers.visual_buffer import ConstrainedVisualReplayBuffer
from pud.runners.runner_habitat_lag import train_eval, eval_pointenv_dists
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
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
    parser.add_argument("--ckpt", type=str, help="policy checkpoint")
    parser.add_argument(
        "--cfg",
        type=str,
        help="Training configuration",
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
        "--encoder", type=str, default="VisualEncoder", help="VisualEncoder"
    )
    parser.add_argument("--eval_interval", type=int, default=-1, help="")
    parser.add_argument("--initial_collect_steps", type=int, default=-1, help="")
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="override cost limit"
    )

    parser.add_argument(
        "--sampler_cost_bounds",
        type=str,
        default="",
        help="cost bounds for pb sampler, delimited by -",
    )
    parser.add_argument(
        "--sampler_dist_bounds",
        type=str,
        default="",
        help="dist bounds for pb sampler, delimited by -",
    )
    parser.add_argument(
        "--sampler_K",
        type=int,
        default=-1,
        help="number of pbs per target, delimited by -",
    )
    parser.add_argument(
        "--sampler_popsize",
        type=int,
        default=-1,
        help="sampler number of random candidate",
    )
    parser.add_argument(
        "--sampler_std_ub",
        type=float,
        default="",
        help="bounds in boosting scores of uncertain samples",
    )
    parser.add_argument(
        "--lambda_lr", type=float, default=-1, help="override lagrange lr"
    )

    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        help="problems that serve as illustration and evaluation set",
    )
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--visual", action="store_true", help="generate and save visual trajs"
    )
    parser.add_argument(
        "--use_disk",
        action="store_true",
        help="use disk if ram is too small for replay buffer",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    args = parser.parse_args()
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    # For dot completion
    cfg = DotMap(cfg)

    if args.lambda_lr > 0:
        cfg.agent_cost_kwargs.lambda_lr = args.lambda_lr
    if args.cost_limit >= 0:
        cfg.agent_cost_kwargs.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    if args.eval_interval > 0:
        cfg.runner.eval_interval = args.eval_interval
    if args.initial_collect_steps > 0:
        cfg.runner.initial_collect_steps = args.initial_collect_steps
    # Override cfs from terminal
    cfg.runner.verbose = args.verbose
    cfg.device = args.device

    if len(args.sampler_cost_bounds) > 0:
        sampler_cost_bounds = [float(x) for x in args.sampler_cost_bounds.split("-")]
        cfg.sampler_extra.cost_bounds = sampler_cost_bounds

    if len(args.sampler_dist_bounds) > 0:
        sampler_dist_bounds = [float(x) for x in args.sampler_dist_bounds.split("-")]
        cfg.sampler.min_dist = sampler_dist_bounds[0]
        cfg.sampler.max_dist = sampler_dist_bounds[1]

    if args.sampler_K > 0:
        cfg.sampler.K = args.sampler_K
    if args.sampler_popsize > 0:
        cfg.sampler.num_states = args.sampler_popsize
    if args.sampler_std_ub > 0:
        cfg.sampler.uncertainty_ub = args.sampler_std_ub

    # Create a lag dir inside the original training dir
    path_ckpt = Path(args.ckpt)
    log_dir = path_ckpt.parent.parent.joinpath("lag")
    cfg.pprint()

    # Custom Logging
    logger = setup_logger(
        root_dir=log_dir.as_posix(),
        subdir_names=["ckpt", "tfevent", "bk", "imgs", "buffer"],
        tag_time=True,
    )

    with open(logger["bk"].joinpath("config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)

    logger["tb"] = SummaryWriter(log_dir=logger["tfevent"].as_posix())  # type: ignore
    logger["pbar"] = args.pbar
    if len(args.illustration_pb_file) > 0:
        logger["illustration_pb_file"] = args.illustration_pb_file

    logger["sampler"] = cfg.sampler
    logger["sampler_extra"] = cfg.sampler_extra

    shutil.copy("launch_jobs/local_train_safe_habitat_lag.sh", logger["bk"].as_posix())
    shutil.copy("launch_jobs/cloud_train_safe_habitat.sh", logger["bk"].as_posix())
    shutil.copy("pud/vision_agent.py", logger["bk"].as_posix())
    shutil.copy("pud/algos/runner_habitat_lag.py", logger["bk"].as_posix())
    shutil.copy("pud/algos/visual_collector.py", logger["bk"].as_posix())

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

    env = safe_habitat_env_load_fn(
        env_kwargs=cfg.env.toDict(),
        cost_f_args=cfg.cost_function.toDict(),
        cost_limit=cfg.agent_cost_kwargs.cost_limit,
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=False,
    )

    set_env_seed(env, cfg.seed)

    eval_env = safe_habitat_env_load_fn(
        env_kwargs=cfg.env.toDict(),
        cost_f_args=cfg.cost_function.toDict(),
        cost_limit=cfg.agent_cost_kwargs.cost_limit,
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + 1)

    assert env.action_space.shape is not None
    cfg.agent["action_dim"] = env.action_space.shape[0]
    cfg.agent["max_action"] = float(env.action_space.high[0])  # type: ignore

    agent = LagVisionUVFDDPG(
        width=cfg.env.simulator_settings.width,
        height=cfg.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder=args.encoder,
        device=cfg.device,
        **cfg.agent.toDict(),
        cost_kwargs=cfg.agent_cost_kwargs.toDict(),
    )

    agent_g = LagVisionUVFDDPG(
        width=cfg.env.simulator_settings.width,
        height=cfg.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder=args.encoder,
        device=cfg.device,
        **cfg.agent.toDict(),
        cost_kwargs=cfg.agent_cost_kwargs.toDict(),
    )

    agent.to(torch.device(args.device))
    agent_g.to(torch.device(args.device))

    agent_g.load_state_dict(torch.load(args.ckpt))
    agent.load_state_dict(torch.load(args.ckpt))

    agent_g.eval()

    print(agent)

    replay_buffer = None
    if args.use_disk:
        from pud.buffers.buffer_large import ConstrainedLargeReplayBuffer

        replay_buffer = ConstrainedLargeReplayBuffer(
            max_size=cfg.replay_buffer.max_size,
            scratch_dir=logger["buffer"].as_posix(),
        )
    else:
        replay_buffer = ConstrainedVisualReplayBuffer(
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

    policy = GaussianPolicy(agent, noise_scale=0.2)

    assert isinstance(replay_buffer, ConstrainedVisualReplayBuffer)
    assert isinstance(env, SafeGoalConditionedHabitatPointQueueWrapper)
    assert isinstance(eval_env, SafeGoalConditionedHabitatPointQueueWrapper)
    train_eval(
        policy=policy,
        agent=agent,
        agent_g=agent_g,
        replay_buffer=replay_buffer,
        env=env,
        eval_env=eval_env,
        eval_func=eval_pointenv_dists,
        pbar=args.pbar,
        logger=logger,
        **cfg.runner,
    )
    torch.save(agent.state_dict(), logger["ckpt"].joinpath("agent.pth"))
