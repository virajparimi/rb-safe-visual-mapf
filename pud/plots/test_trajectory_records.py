import yaml
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap

from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_global_seed, set_env_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.policies import ConstrainedMultiAgentSearchPolicy
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.pb_sampler import load_pb_set, sample_pbs_by_agent
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
    SafeGoalConditionedPointQueueWrapper,
)


def pointenv_setup(args):
    assert len(args.config_file) > 0
    assert len(args.constrained_ckpt_file) > 0

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # User defined parameters for evaluation
    trained_cost_limit = config.agent.cost_limit

    config.device = args.device
    config.num_samples = args.num_samples
    config.replay_buffer.max_size = args.replay_buffer_size

    set_global_seed(config.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in config.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

    eval_env = safe_env_load_fn(
        config.env.toDict(),
        config.cost_function.toDict(),
        max_episode_steps=config.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )

    set_env_seed(eval_env, config.seed + 2)

    obs_dim = eval_env.observation_space["observation"].shape[0]  # type: ignore
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    logging.debug(
        f"Obs dim: {obs_dim},\n"
        f"Goal dim: {goal_dim},\n"
        f"State dim: {state_dim},\n"
        f"Action dim: {action_dim},\n"
        "Max action: {max_action}"
    )

    agent = DRLDDPGLag(
        state_dim,  # Concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(config.device),
        **config.agent,
    )

    agent.load_state_dict(
        torch.load(args.constrained_ckpt_file, map_location=torch.device(config.device))
    )
    agent.to(torch.device(config.device))
    agent.eval()

    return config, eval_env, agent, trained_cost_limit


def load_agent_and_env(agent, eval_env, args, config, constrained=False):
    if constrained:
        agent.load_state_dict(
            torch.load(
                args.constrained_ckpt_file, map_location=torch.device(config.device)
            )
        )
    else:
        agent.load_state_dict(
            torch.load(
                args.unconstrained_ckpt_file, map_location=torch.device(config.device)
            )
        )
    agent.to(torch.device(config.device))
    agent.eval()

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    return agent, eval_env


def setup_problems(eval_env, agent, args, config, basedir):

    rb_vec = ConstrainedCollector.sample_initial_unconstrained_states(
        eval_env, config.replay_buffer.max_size, habitat=False
    )

    agent.load_state_dict(
        torch.load(
            args.unconstrained_ckpt_file, map_location=torch.device(config.device)
        )
    )
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore
    unconstrained_pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore

    if len(args.illustration_pb_file) > 0:
        problems = load_pb_set(file_path=args.illustration_pb_file, env=eval_env, agent=agent)  # type: ignore
    else:
        K = 5
        difficulty = eval_env.max_goal_dist
        if args.traj_difficulty == "easy":
            difficulty = eval_env.max_goal_dist // 8
        elif args.traj_difficulty == "medium":
            difficulty = eval_env.max_goal_dist // 4
        elif args.traj_difficulty == "hard":
            difficulty = eval_env.max_goal_dist // 2
        problems = []
        for _ in tqdm(range(config.num_samples * args.num_agents // K)):
            inter_problems = sample_pbs_by_agent(
                K=K,
                min_dist=0,
                max_dist=eval_env.max_goal_dist,  # type: ignore
                target_val=difficulty,  # type: ignore
                agent=agent,  # type: ignore
                env=eval_env,  # type: ignore
                num_states=1000,
                ensemble_agg="mean",
                use_uncertainty=False,
            )
            assert len(inter_problems) > 0
            problems.extend(inter_problems)

    agent.load_state_dict(
        torch.load(args.constrained_ckpt_file, map_location=torch.device(config.device))
    )
    constrained_pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore

    if "unconstrained" in args.method_type:
        agent.load_state_dict(
            torch.load(
                args.unconstrained_ckpt_file, map_location=torch.device(config.device)
            )
        )

    return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems


def load_problem_set(file_path, env, agent):
    load = np.load(file_path, allow_pickle=True)
    rb_vec = load["rb_vec"]
    unconstrained_pdist = load["unconstrained_pdist"]
    constrained_pdist = load["constrained_pdist"]
    pcost = load["pcost"]
    problems = load["problems"]
    return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems.tolist()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--visual", default=False, action="store_true")
    parser.add_argument("--illustration_pb_file", type=str, default="")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument("--collect_trajs", default=False, action="store_true")
    parser.add_argument("--load_problem_set", default=False, action="store_true")
    parser.add_argument("--use_unconstrained_ckpt", default=False, action="store_true")
    parser.add_argument(
        "--traj_difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--method_type",
        type=str,
        choices=["risk_bounded_constrained_search_ds"],
        default="risk_bounded_constrained_search_ds",
    )

    args = parser.parse_args()
    return args


def multi_constrained_search_policy(agent, eval_env, problem_setup, args, config, trained_cost_limit, basedir):
    if args.use_unconstrained_ckpt:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=False
        )
    else:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=True
        )

    rb_vec = problem_setup[0].copy()
    pdist = problem_setup[2].copy()
    pcost = problem_setup[3].copy()
    problems = problem_setup[-1].copy()

    factor = 1.0
    logging.info(f"Factor: {factor}")

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    edge_cost_limit = trained_cost_limit * factor

    constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
        agent,
        rb_vec.copy(),
        args.num_agents,
        radius=0.0,
        open_loop=True,
        risk_bound=trained_cost_limit,
        disjoint_split=True,
        pdist=pdist.copy(),
        pcost=pcost.copy(),
        no_waypoint_hopping=True,
        max_cost_limit=edge_cost_limit,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    for _ in tqdm(range(config.num_samples)):
        _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
            constrained_ma_search_policy,
            eval_env,
            args.num_agents,
            threshold=0.0,
            habitat=False,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = argument_parser()
    config, eval_env, agent, trained_cost_limit = pointenv_setup(args)

    basedir = Path("pud/plots/data")
    basedir = basedir / config.env.walls.lower()

    if not basedir.exists():
        basedir.mkdir(parents=True)

    assert args.collect_trajs is False
    assert args.load_problem_set
    assert len(args.problem_set_file) > 0
    problem_setup = load_problem_set(args.problem_set_file, eval_env, agent)

    assert args.num_agents != 1
    multi_constrained_search_policy(agent, eval_env, problem_setup, args, config, trained_cost_limit, basedir)
