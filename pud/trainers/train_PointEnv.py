import yaml
import torch
import argparse
from pathlib import Path
from dotmap import DotMap
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import GaussianPolicy
from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.runners.crl_runner_v3 import train_eval, eval_pointenv_cost_constrained_dists
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
    safe_env_load_fn,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="Training configuration",
    )
    parser.add_argument(
        "--env", type=str, default="", help="terminal override of training env"
    )
    parser.add_argument(
        "--resize_factor", type=int, default=-1, help="override default resize factor"
    )
    parser.add_argument(
        "--action_noise", type=float, default=-1, help="action noise from env"
    )
    parser.add_argument("--actor_lr", type=float, default=-1, help="actor lr")
    parser.add_argument("--critic_lr", type=float, default=-1, help="critic lr")
    parser.add_argument("--cost_name", type=str, default="", help="override cost type")
    parser.add_argument(
        "--cost_radius", type=float, default=-1, help="override cost type"
    )
    parser.add_argument("--cost_max", type=float, default=-1, help="override if > 0")
    parser.add_argument("--cost_N", type=int, default=0, help="override if > 0")
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

    # Override cfs from terminal
    if len(args.env) > 0:
        cfg.env.walls = args.env
    if args.resize_factor > 0:
        cfg.env.resize_factor = args.resize_factor
    if args.action_noise > 0:
        cfg.env.action_noise = args.action_noise
    if args.actor_lr > 0:
        cfg.agent.actor_lr = args.actor_lr
    if args.critic_lr > 0:
        cfg.agent.critic_lr = args.critic_lr
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
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    if len(args.logdir) > 0:
        cfg.ckpt_dir = args.logdir
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
    if len(args.ckpt) > 0:
        assert Path(args.ckpt).exists()
        print("[INFO] loading checkpoint: {}".format(args.ckpt))
        agent.load_state_dict(torch.load(args.ckpt))
    agent.to(torch.device(args.device))

    print(agent)

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )

    # Custom logging
    log_dir = Path(cfg.ckpt_dir)
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

    tb = SummaryWriter(log_dir=tfevent_dir.as_posix())

    # Gaussian policy seems just add exploration noise, the evaluation code
    # does not use it
    policy = GaussianPolicy(agent)

    turn_on_lag = False

    train_eval(
        policy,
        agent,
        replay_buffer,
        env,
        eval_env,
        eval_func=eval_pointenv_cost_constrained_dists,
        tensorboard_writer=tb,
        pbar=args.pbar,
        ckpt_dir=ckpt_dir,
        vis_dir=vis_dir,
        i_start=args.i_start,
        turn_on_lag=turn_on_lag,
        illustration_pb_file=args.illustration_pb_file,
        **cfg.runner,
    )
    torch.save(
        agent.state_dict(),
        ckpt_dir.joinpath("agent.pth"),
    )
