"""
Load and visualize trained policy on manually crafted illustration problems
"""

import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from copy import deepcopy
from gym.spaces import Box
import matplotlib.pyplot as plt


from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.pb_sampler import load_pb_set
from pud.visualizers.visualize import visualize_eval_records
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import eval_agent_from_Q

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
    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        default="",
        help="path to illustration reference problem",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
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
    args = parser.parse_args()

    setup_ret = setup_env(args)
    agent = setup_ret["agent"]
    eval_env = setup_ret["eval_env"]  # type: ignore
    cfg = setup_ret["cfg"]
    obs_dim = setup_ret["obs_dim"]
    goal_dim = setup_ret["goal_dim"]
    state_dim = setup_ret["state_dim"]
    action_dim = setup_ret["action_dim"]
    max_action = setup_ret["max_action"]
    replay_buffer = setup_ret["replay_buffer"]
    figdir = setup_ret["figsavedir"]

    eval_env.set_sample_goal_args(
        prob_constraint=0.0,
        min_dist=0,
        max_dist=np.inf,
        min_cost=0.0,
        max_cost=1.0,
    )

    eval_env: SafeGoalConditionedPointQueueWrapper
    if len(args.illustration_pb_file) > 0:
        assert isinstance(agent, DRLDDPGLag)
        assert isinstance(eval_env, SafeGoalConditionedPointQueueWrapper)
        cost_eval_pbs = load_pb_set(
            file_path=args.illustration_pb_file,
            env=eval_env,
            agent=agent,
        )
        agent.load_state_dict(torch.load(args.ckpt))
        agent.eval()
        collect_trajs = True
        eval_env.set_pbs(pb_list=[tuple(pb.items()) for pb in deepcopy(cost_eval_pbs)])
        cost_eval_i = eval_agent_from_Q(
            policy=agent,
            eval_env=eval_env,
            collect_trajs=collect_trajs,
        )
        if collect_trajs:
            start_list = [p["start"].tolist() for p in cost_eval_pbs]
            goal_list = [p["goal"].tolist() for p in cost_eval_pbs]
            fig, ax = plt.subplots()
            ax = visualize_eval_records(
                eval_records=cost_eval_i,
                eval_env=eval_env,
                ax=ax,
                starts=start_list,
                goals=goal_list,
                color="g",
            )
            for ii in range(len(start_list)):
                xy_n = eval_env.normalize_obs(start_list[ii])
                ax.text(
                    x=xy_n[0] + 0.1,
                    y=xy_n[1],
                    s="cost={:.2f}, predicted cost={:.2f}".format(
                        cost_eval_i[ii]["cum_costs"],
                        cost_eval_pbs[ii]["info"]["prediction"],
                    ),
                )
            ax.set_title("illustration problems")
            ax.legend()
            figname = "ref.jpg"
            assert isinstance(figdir, Path)
            fig.savefig(figdir.joinpath(figname), dpi=300)
            plt.close(fig=fig)
