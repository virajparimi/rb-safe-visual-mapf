"""
This is the 2nd stage of policy training, it takes a policy without Lagrange Penalty and train a new policy with
Lagrange Multiplier to respect cost constraints
"""

import yaml
import torch
import shutil
import argparse
from pathlib import Path
from dotmap import DotMap
from torch.utils.tensorboard.writer import SummaryWriter


from pud.algos.policies import GaussianPolicy
from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.runners.runner_lag import train_eval, eval_pointenv_cost_constrained_dists
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)

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
    parser.add_argument("--eval_interval", type=int, default=-1, help="")
    parser.add_argument("--initial_collect_steps", type=int, default=-1, help="")
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="override cost limit"
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
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    args = parser.parse_args()
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    # For dot completion
    cfg = DotMap(cfg)

    if args.lambda_lr > 0:
        cfg.agent.lambda_lr = args.lambda_lr
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
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
    cfg.pprint()

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=False,
    )
    set_env_seed(env, cfg.seed + 1)

    eval_env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = env.observation_space["observation"].shape[0]  # type: ignore

    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim

    assert env.action_space.shape is not None
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # type: ignore
    print(
        f"Observation dimension: {obs_dim},\n"
        f"Goal dimension: {goal_dim},\n"
        f"State dimension: {state_dim},\n"
        f"Action dimension: {action_dim},\n"
        f"Max Action: {max_action}"
    )

    agent = DRLDDPGLag(
        # DDPG args
        state_dim,  # concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
    )
    agent_g = DRLDDPGLag(
        # DDPG args
        state_dim,  # concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
    )
    agent.to(torch.device(args.device))
    agent_g.to(torch.device(args.device))

    agent.load_state_dict(torch.load(args.ckpt))
    agent_g.load_state_dict(torch.load(args.ckpt))

    # Check lambda_lr
    if args.lambda_lr > 0:
        assert (
            agent.lagrange.lambda_optimizer.state_dict()["param_groups"][0]["lr"]
            == args.lambda_lr
        )

    agent_g.eval()
    print(agent)

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )

    # Create a lag dir inside the original training dir
    path_ckpt = Path(args.ckpt)
    log_dir = path_ckpt.parent.parent.joinpath("lag")
    from datetime import datetime

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = log_dir.joinpath(date_time)
    ckpt_dir = log_dir.joinpath("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tfevent_dir = log_dir.joinpath("tfevent")
    tfevent_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None
    if args.visual:
        vis_dir = log_dir.joinpath("visual")
        vis_dir.mkdir(parents=True, exist_ok=True)
    bk_dir = log_dir.joinpath("bk")
    bk_dir.mkdir(parents=True, exist_ok=True)
    with open(bk_dir.joinpath("bk_config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    shutil.copy("pud/algos/lagrange/drl_ddpg_lag.py", bk_dir.as_posix())
    shutil.copy("launch_jobs/local_lag_train.sh", bk_dir.as_posix())
    shutil.copy("pud/algos/runner_lag.py", bk_dir.as_posix())

    tb = SummaryWriter(log_dir=tfevent_dir.as_posix())

    # Gaussian policy seems just add exploration noise, the evaluation code
    # does not use it
    policy = GaussianPolicy(agent)

    train_eval(
        policy,
        agent,
        agent_g,
        replay_buffer,
        env,
        eval_env,
        eval_func=eval_pointenv_cost_constrained_dists,
        tensorboard_writer=tb,
        pbar=args.pbar,
        illustration_pb_file=args.illustration_pb_file,
        ckpt_dir=ckpt_dir,
        vis_dir=vis_dir,
        **cfg.runner,
    )
    torch.save(
        agent.state_dict(),
        ckpt_dir.joinpath("agent.pth"),
    )
