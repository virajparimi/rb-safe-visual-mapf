import yaml
import torch
import argparse
from pathlib import Path
from dotmap import DotMap
from gym.spaces import Box
import matplotlib.pyplot as plt


from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.visualizers.visualize import visualize_eval_records
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import eval_agent_from_Q
from pud.envs.safe_pointenv.pb_sampler import (
    sample_pbs_by_agent,
    sample_cost_pbs_by_agent,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)


def setup_args_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="training configuration",
    )
    parser.add_argument("--figsavedir", type=str, help="directory to save figures")
    parser.add_argument("--ckpt", type=str, help="the path to ckpt file")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="show progress bar")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose printing/logging"
    )
    return parser


def setup_env(args: argparse.Namespace):
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)

    cfg.runner.verbose = args.verbose
    cfg.device = args.device

    cfg.pprint()

    figdir = Path(args.figsavedir)
    figdir.mkdir(parents=True, exist_ok=True)

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

    eval_env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    assert (
        isinstance(eval_env, SafeGoalConditionedPointWrapper)
        or isinstance(eval_env, SafeGoalConditionedPointQueueWrapper)
        or isinstance(eval_env, SafeGoalConditionedPointBlendWrapper)
    )
    eval_env.set_prob_constraint(1.0)

    set_env_seed(eval_env, cfg.seed + 2)

    assert isinstance(eval_env.observation_space, dict)
    obs_dim = eval_env.observation_space["observation"].shape[0]
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    assert eval_env.action_space.shape is not None
    action_dim = eval_env.action_space.shape[0]
    assert isinstance(eval_env.action_space, Box)
    max_action = float(eval_env.action_space.high[0])
    print(
        f"Obs dim: {obs_dim}, Goal dim: {goal_dim}, State dim: {state_dim}, "
        f"Action dim: {action_dim}, Max action: {max_action}"
    )

    agent = DRLDDPGLag(
        # DDPG args
        state_dim,  # Concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
    )

    ckpt_file = args.ckpt
    agent.load_state_dict(torch.load(ckpt_file))
    agent.eval()

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )
    return dict(
        agent=agent,
        eval_env=eval_env,
        figsavedir=figdir,
        cfg=cfg,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        replay_buffer=replay_buffer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_args_parser(parser)
    parser.add_argument("--K", type=int, default=10, help="num of problems")
    parser.add_argument("--N", type=int, default=50, help="sample size")
    parser.add_argument("--metric", type=str, default="cost", help="cost | dist")
    parser.add_argument(
        "--target", type=float, default=None, help="target cost or distance"
    )
    parser.add_argument("--min_dist", type=float, default=0, help="minimum distance")
    parser.add_argument("--max_dist", type=float, default=10, help="maximum distance")
    parser.add_argument(
        "--figname", type=str, default="pb_samples.jpg", help="figure name"
    )
    args = parser.parse_args()

    setup_ret = setup_env(args)
    agent = setup_ret["agent"]
    eval_env = setup_ret["eval_env"]
    cfg = setup_ret["cfg"]
    obs_dim = setup_ret["obs_dim"]
    goal_dim = setup_ret["goal_dim"]
    state_dim = setup_ret["state_dim"]
    action_dim = setup_ret["action_dim"]
    max_action = setup_ret["max_action"]
    replay_buffer = setup_ret["replay_buffer"]
    figdir = setup_ret["figsavedir"]

    # Rollout trained policy and visualize trajectory
    assert (
        isinstance(eval_env, SafeGoalConditionedPointWrapper)
        or isinstance(eval_env, SafeGoalConditionedPointQueueWrapper)
        or isinstance(eval_env, SafeGoalConditionedPointBlendWrapper)
    )
    eval_env.set_prob_constraint(1.0)

    pbs = []
    assert isinstance(agent, DRLDDPGLag)
    if args.metric == "dist":
        pbs = sample_pbs_by_agent(
            env=eval_env,
            agent=agent,
            num_states=args.N,
            target_val=args.target,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            ensemble_agg="mean",
            K=args.K,
        )
    elif args.metric == "cost":
        pbs = sample_cost_pbs_by_agent(
            env=eval_env,
            agent=agent,
            num_states=args.N,
            target_val=args.target,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            ensemble_agg="mean",
            K=args.K,
        )
    else:
        raise Exception("metric is incorrect")

    pb_tuples = [(p["start"], p["goal"], p["info"]) for p in pbs]
    eval_env.set_pbs(pb_list=pb_tuples)  # type: ignore
    start_list = [p["start"].tolist() for p in pbs]
    goal_list = [p["goal"].tolist() for p in pbs]

    eval_records = eval_agent_from_Q(
        policy=agent,
        eval_env=eval_env,
        collect_trajs=True,
    )

    fig, ax = plt.subplots()
    ax = visualize_eval_records(
        eval_records=eval_records,
        eval_env=eval_env,
        ax=ax,
        starts=start_list,
        goals=goal_list,
        use_pbar=True,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    assert isinstance(figdir, Path)
    fig.savefig(figdir.joinpath(args.figname), dpi=300)
    plt.close(fig)
