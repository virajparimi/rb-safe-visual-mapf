import sys
import yaml
import torch
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap

from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_global_seed, set_env_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_pointenv.pb_sampler import load_pb_set, sample_pbs_by_agent
from pud.algos.policies import (
    SearchPolicy,
    VisualSearchPolicy,
    MultiAgentSearchPolicy,
    ConstrainedSearchPolicy,
    VisualMultiAgentSearchPolicy,
    VisualConstrainedSearchPolicy,
    ConstrainedMultiAgentSearchPolicy,
    VisualConstrainedMultiAgentSearchPolicy,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
    SafeGoalConditionedPointQueueWrapper,
)
from pud.algos.vision.vision_agent import LagVisionUVFDDPG


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
    action_dim = eval_env.action_space.shape[0]  # type: ignore
    max_action = float(eval_env.action_space.high[0])  # type: ignore
    logging.debug(
        f"Obs dim: {obs_dim},\n"
        f"Goal dim: {goal_dim},\n"
        f"State dim: {state_dim},\n"
        f"Action dim: {action_dim},\n"
        f"Max action: {max_action}"
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


def habitat_setup(args):
    assert len(args.config_file) > 0
    assert len(args.constrained_ckpt_file) > 0

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # User defined parameters for evaluation
    trained_cost_limit = config.agent_cost_kwargs.cost_limit

    config.device = args.device
    config.num_samples = args.num_samples
    config.replay_buffer.max_size = args.replay_buffer_size

    set_global_seed(config.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in config.wrappers:
        if wrapper_name == "GoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(GoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

    eval_env = safe_habitat_env_load_fn(
        env_kwargs=config.env.toDict(),
        cost_f_args=config.cost_function.toDict(),
        cost_limit=config.agent_cost_kwargs.cost_limit,
        max_episode_steps=config.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, config.seed + 1)

    config.agent["action_dim"] = eval_env.action_space.shape[0]  # type: ignore
    config.agent["max_action"] = float(eval_env.action_space.high[0])  # type: ignore

    agent = LagVisionUVFDDPG(
        width=config.env.simulator_settings.width,
        height=config.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder="VisualEncoder",
        device=config.device,
        **config.agent.toDict(),
        cost_kwargs=config.agent_cost_kwargs.toDict(),
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


def setup_problems(eval_env, agent, args, config, basedir, save=False):

    habitat = args.visual
    rb_vec = ConstrainedCollector.sample_initial_unconstrained_states(
        eval_env, config.replay_buffer.max_size, habitat=habitat
    )

    if habitat:
        rb_vec_grid, rb_vec = rb_vec

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
        print(len(problems))

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

    if save:
        if args.traj_difficulty == "easy":
            save_path = basedir / "easy.npz"
        elif args.traj_difficulty == "medium":
            save_path = basedir / "medium.npz"
        elif args.traj_difficulty == "hard":
            save_path = basedir / "hard.npz"
        if not habitat:
            np.savez(
                save_path,
                rb_vec=rb_vec,
                unconstrained_pdist=unconstrained_pdist,
                constrained_pdist=constrained_pdist,
                pcost=pcost,
                problems=problems,  # type: ignore
            )
        else:
            np.savez(
                save_path,
                rb_vec_grid=rb_vec_grid,
                rb_vec=rb_vec,
                unconstrained_pdist=unconstrained_pdist,
                constrained_pdist=constrained_pdist,
                pcost=pcost,
                problems=problems,  # type: ignore
            )

    if not habitat:
        return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems
    else:
        return (
            rb_vec_grid,
            rb_vec,
            unconstrained_pdist,
            constrained_pdist,
            pcost,
            problems,
        )


def load_problem_set(file_path, env, agent, habitat=False):
    load = np.load(file_path, allow_pickle=True)
    if habitat:
        rb_vec_grid = load["rb_vec_grid"]
    rb_vec = load["rb_vec"]
    unconstrained_pdist = load["unconstrained_pdist"]
    constrained_pdist = load["constrained_pdist"]
    pcost = load["pcost"]
    problems = load["problems"]
    if not habitat:
        return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems.tolist()
    else:
        return (
            rb_vec_grid,
            rb_vec,
            unconstrained_pdist,
            constrained_pdist,
            pcost,
            problems.tolist(),
        )


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
        choices=[
            "unconstrained",
            "unconstrained_search",
            "constrained",
            "constrained_search",
            "unconstrained_search_ds",
            "constrained_search_ds",
            "risk_bounded_constrained_search_ds",
        ],
        default="unconstrained",
    )

    args = parser.parse_args()
    return args


def single_unconstrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_records = []
    save_path = basedir / "single_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / "unconstrained_records.npy"
    if save and Path(save_path).exists():
        unconstrained_records = np.load(save_path, allow_pickle=True)
        unconstrained_records = unconstrained_records.tolist()

    start_idx = len(unconstrained_records)
    logging.info(f"Starting from index: {start_idx}")

    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    for _ in range(start_idx, config.num_samples):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(
                agent, eval_env, habitat=habitat
            )
            unconstrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            unconstrained_records.append({})

        if save:
            np.save(save_path, unconstrained_records)

    if save:
        np.save(save_path, unconstrained_records)
    return unconstrained_records


def multi_unconstrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_records = []
    save_path = basedir / "multi_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"unconstrained_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        unconstrained_records = np.load(save_path, allow_pickle=True)
        unconstrained_records = unconstrained_records.tolist()

    start_idx = len(unconstrained_records) * args.num_agents
    logging.info(f"Starting from index: {start_idx // args.num_agents}")

    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    for _ in tqdm(range(start_idx // args.num_agents, config.num_samples)):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                agent, eval_env, args.num_agents, threshold=0.0, habitat=habitat
            )
            unconstrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            unconstrained_records.append([{} for _ in range(args.num_agents)])

        if save:
            np.save(save_path, unconstrained_records)

    if save:
        np.save(save_path, unconstrained_records)
    return unconstrained_records


def single_unconstrained_search_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_search_records = []
    save_path = basedir / "single_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / "unconstrained_search_records.npy"
    if save and Path(save_path).exists():
        unconstrained_search_records = np.load(save_path, allow_pickle=True)
        unconstrained_search_records = unconstrained_search_records.tolist()

    start_idx = len(unconstrained_search_records)
    logging.info(f"Starting from index: {start_idx}")

    if not habitat:
        rb_vec, pdist = problem_setup[0].copy(), problem_setup[1].copy()
    else:
        rb_vec_grid, rb_vec, pdist = (
            problem_setup[0].copy(),
            problem_setup[1].copy(),
            problem_setup[2].copy(),
        )
    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    if not habitat:
        search_policy = SearchPolicy(
            agent, rb_vec, pdist=pdist, open_loop=True, no_waypoint_hopping=True
        )
    else:
        search_policy = VisualSearchPolicy(
            agent,
            (rb_vec_grid, rb_vec),
            pdist=pdist,
            open_loop=True,
            max_search_steps=4,
            no_waypoint_hopping=True,
        )

    for _ in tqdm(range(start_idx, config.num_samples)):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(
                search_policy, eval_env, habitat=habitat
            )
            unconstrained_search_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            unconstrained_search_records.append({})

        if save:
            np.save(save_path, unconstrained_search_records)

    if save:
        np.save(save_path, unconstrained_search_records)
    return unconstrained_search_records


def multi_unconstrained_search_policy(
    agent, eval_env, problem_setup, args, config, basedir, ds=False, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_search_records = []
    save_path = basedir / "multi_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if ds:
        save_path = save_path / f"unconstrained_search_ds_records_{args.num_agents}.npy"
    else:
        save_path = save_path / f"unconstrained_search_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        unconstrained_search_records = np.load(save_path, allow_pickle=True)
        unconstrained_search_records = unconstrained_search_records.tolist()

    start_idx = len(unconstrained_search_records) * args.num_agents
    logging.info(f"Starting from index: {start_idx // args.num_agents}")

    if not habitat:
        rb_vec, pdist = problem_setup[0].copy(), problem_setup[1].copy()
    else:
        rb_vec_grid, rb_vec, pdist = (
            problem_setup[0].copy(),
            problem_setup[1].copy(),
            problem_setup[2].copy(),
        )
    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    if not habitat:
        ma_search_policy = MultiAgentSearchPolicy(
            agent,
            rb_vec,
            args.num_agents,
            pdist=pdist,
            open_loop=True,
            no_waypoint_hopping=True,
            radius=0,
            disjoint_split=ds,
        )
    else:
        ma_search_policy = VisualMultiAgentSearchPolicy(
            agent,
            (rb_vec_grid, rb_vec),
            args.num_agents,
            pdist=pdist,
            open_loop=True,
            disjoint_split=ds,
            max_search_steps=4,
            no_waypoint_hopping=True,
        )

    for _ in tqdm(range(start_idx // args.num_agents, config.num_samples)):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                ma_search_policy,
                eval_env,
                args.num_agents,
                threshold=0.0,
                habitat=habitat,
            )
            unconstrained_search_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            unconstrained_search_records.append([{} for _ in range(args.num_agents)])

        if save:
            np.save(save_path, unconstrained_search_records)

    if save:
        np.save(save_path, unconstrained_search_records)
    return unconstrained_search_records


def single_constrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    constrained_records = []
    save_path = basedir / "single_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / "constrained_records.npy"
    if save and Path(save_path).exists():
        constrained_records = np.load(save_path, allow_pickle=True)
        constrained_records = constrained_records.tolist()

    start_idx = len(constrained_records)
    logging.info(f"Starting from index: {start_idx}")

    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    for _ in range(start_idx, config.num_samples):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(
                agent, eval_env, habitat=habitat
            )
            constrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            constrained_records.append({})

        if save:
            np.save(save_path, constrained_records)

    if save:
        np.save(save_path, constrained_records)
    return constrained_records


def multi_constrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    constrained_records = []
    save_path = basedir / "multi_agent" / args.traj_difficulty
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"constrained_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        constrained_records = np.load(save_path, allow_pickle=True)
        constrained_records = constrained_records.tolist()

    start_idx = len(constrained_records) * args.num_agents
    logging.info(f"Starting from index: {start_idx // args.num_agents}")

    problems = problem_setup[-1].copy()
    problems = problems[start_idx:]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    for _ in tqdm(range(start_idx // args.num_agents, config.num_samples)):
        try:
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                agent, eval_env, args.num_agents, threshold=0.0, habitat=habitat
            )
            constrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            constrained_records.append([{} for _ in range(args.num_agents)])

        if save:
            np.save(save_path, constrained_records)

    if save:
        np.save(save_path, constrained_records)
    return constrained_records


def single_constrained_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    trained_cost_limit,
    basedir,
    save=False,
):
    habitat = args.visual
    if args.use_unconstrained_ckpt:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=False
        )
    else:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=True
        )

    if not habitat:
        rb_vec = problem_setup[0].copy()
        pdist = problem_setup[2].copy()
        pcost = problem_setup[3].copy()
    else:
        rb_vec_grid = problem_setup[0].copy()
        rb_vec = problem_setup[1].copy()
        pdist = problem_setup[3].copy()
        pcost = problem_setup[4].copy()
    problems = problem_setup[-1].copy()

    eval_env.set_prob_constraint(1.0)  # type: ignore

    constrained_search_factored_records = []
    edge_cost_limit_factors = [0.1, 0.25, 0.5, 0.75, 1.0]
    for factor in edge_cost_limit_factors:
        logging.info(f"Factor: {factor}")

        constrained_search_records = []
        save_path = basedir / "single_agent" / args.traj_difficulty
        if not save_path.exists():
            save_path.mkdir(parents=True)
        save_path = save_path / f"constrained_search_records_{factor}.npy"
        if args.use_unconstrained_ckpt:
            save_path = save_path.as_posix()[:-4] + "_uc.npy"
        if save and Path(save_path).exists():
            constrained_search_records = np.load(save_path, allow_pickle=True)
            constrained_search_records = constrained_search_records.tolist()

        start_idx = len(constrained_search_records)
        logging.info(f"Starting from index: {start_idx}")

        problems = problem_setup[-1].copy()
        problems = problems[start_idx:]
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        edge_cost_limit = trained_cost_limit * factor

        if not habitat:
            constrained_search_policy = ConstrainedSearchPolicy(
                agent,
                rb_vec,
                pdist=pdist,
                pcost=pcost,
                open_loop=True,
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit,
                ckpts={
                    "unconstrained": args.unconstrained_ckpt_file,
                    "constrained": args.constrained_ckpt_file,
                },
            )
        else:
            constrained_search_policy = VisualConstrainedSearchPolicy(
                agent,
                (rb_vec_grid, rb_vec),
                pdist=pdist,
                pcost=pcost,
                open_loop=True,
                max_search_steps=4,
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit,
                ckpts={
                    "unconstrained": args.unconstrained_ckpt_file,
                    "constrained": args.constrained_ckpt_file,
                },
            )

        for _ in tqdm(range(start_idx, config.num_samples)):
            try:
                _, _, _, _, _, records = ConstrainedCollector.get_trajectory(
                    constrained_search_policy, eval_env, habitat=habitat
                )
                constrained_search_records.append(records)
            except Exception as e:
                logging.error(f"Error: {e}")
                constrained_search_records.append({})

            if save:
                np.save(save_path, constrained_search_records)

        if save:
            np.save(save_path, constrained_search_records)

        constrained_search_factored_records.append(constrained_search_records)

    if save:
        save_path = basedir / "single_agent" / args.traj_difficulty
        if not save_path.exists():
            save_path.mkdir(parents=True)
        save_path = save_path / "constrained_search_factored_records.npy"
        if args.use_unconstrained_ckpt:
            save_path = save_path.as_posix()[:-4] + "_uc.npy"
        np.save(save_path, constrained_search_factored_records)
    return constrained_search_factored_records


def multi_constrained_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    trained_cost_limit,
    basedir,
    ds=False,
    risk_bounded=False,
    save=False,
):
    habitat = args.visual
    if args.use_unconstrained_ckpt:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=False
        )
    else:
        agent, eval_env = load_agent_and_env(
            agent, eval_env, args, config, constrained=True
        )

    if not habitat:
        rb_vec = problem_setup[0].copy()
        pdist = problem_setup[2].copy()
        pcost = problem_setup[3].copy()
    else:
        rb_vec_grid = problem_setup[0].copy()
        rb_vec = problem_setup[1].copy()
        pdist = problem_setup[3].copy()
        pcost = problem_setup[4].copy()
    problems = problem_setup[-1].copy()

    constrained_search_factored_records = []
    edge_cost_limit_factors = [0.1, 0.25, 0.5, 0.75, 1.0]
    for factor in edge_cost_limit_factors:
        logging.info(f"Factor: {factor}")

        constrained_search_records = []
        save_path = basedir / "multi_agent" / args.traj_difficulty
        if not save_path.exists():
            save_path.mkdir(parents=True)

        if ds and not risk_bounded:
            save_path = (
                save_path
                / f"constrained_search_ds_records_{args.num_agents}_{factor}.npy"
            )
        elif ds and risk_bounded:
            save_path = (
                save_path
                / f"risk_bounded_constrained_search_ds_records_{args.num_agents}_{factor}.npy"
            )
        else:
            save_path = (
                save_path / f"constrained_search_records_{args.num_agents}_{factor}.npy"
            )

        if args.use_unconstrained_ckpt:
            save_path = save_path.as_posix()[:-4] + "_uc.npy"
        if save and Path(save_path).exists():
            constrained_search_records = np.load(save_path, allow_pickle=True)
            constrained_search_records = constrained_search_records.tolist()

        start_idx = len(constrained_search_records) * args.num_agents
        logging.info(f"Starting from index: {start_idx // args.num_agents}")

        problems = problem_setup[-1].copy()
        problems = problems[start_idx:]
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        edge_cost_limit = trained_cost_limit * factor

        if not habitat:
            constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
                agent,
                rb_vec.copy(),
                args.num_agents,
                radius=0.0,
                open_loop=True,
                risk_bound=(
                    edge_cost_limit * args.num_agents if risk_bounded else np.inf
                ),
                disjoint_split=ds,
                pdist=pdist.copy(),
                pcost=pcost.copy(),
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit,
                ckpts={
                    "unconstrained": args.unconstrained_ckpt_file,
                    "constrained": args.constrained_ckpt_file,
                },
            )
        else:
            constrained_ma_search_policy = VisualConstrainedMultiAgentSearchPolicy(
                agent,
                (rb_vec_grid.copy(), rb_vec.copy()),
                args.num_agents,
                radius=0.0,
                pdist=pdist.copy(),
                pcost=pcost.copy(),
                open_loop=True,
                risk_bound=(
                    edge_cost_limit * args.num_agents if risk_bounded else np.inf
                ),
                disjoint_split=ds,
                max_search_steps=4,
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit,
                ckpts={
                    "unconstrained": args.unconstrained_ckpt_file,
                    "constrained": args.constrained_ckpt_file,
                },
            )

        for _ in tqdm(range(start_idx // args.num_agents, config.num_samples)):
            try:
                _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                    constrained_ma_search_policy,
                    eval_env,
                    args.num_agents,
                    threshold=0.0,
                    habitat=habitat,
                )
                constrained_search_records.append(records)
            except Exception as e:
                logging.error(f"Error: {e}")
                constrained_search_records.append([{} for _ in range(args.num_agents)])

            if save:
                np.save(save_path, constrained_search_records)

        if save:
            np.save(save_path, constrained_search_records)

        constrained_search_factored_records.append(constrained_search_records)

    if save:
        save_path = basedir / "multi_agent" / args.traj_difficulty
        if not save_path.exists():
            save_path.mkdir(parents=True)

        if ds and not risk_bounded:
            save_path = (
                save_path
                / f"constrained_search_ds_factored_records_{args.num_agents}.npy"
            )
        elif ds and risk_bounded:
            save_path = (
                save_path
                / f"risk_bounded_constrained_search_ds_factored_records_{args.num_agents}.npy"
            )
        else:
            save_path = (
                save_path / f"constrained_search_factored_records_{args.num_agents}.npy"
            )
        if args.use_unconstrained_ckpt:
            save_path = save_path.as_posix()[:-4] + "_uc.npy"
        np.save(save_path, constrained_search_factored_records)
    return constrained_search_factored_records


def main():
    save = False
    args = argument_parser()
    if args.visual:
        config, eval_env, agent, trained_cost_limit = habitat_setup(args)
    else:
        config, eval_env, agent, trained_cost_limit = pointenv_setup(args)

    basedir = Path("pud/plots/data")
    if not args.visual:
        basedir = basedir / config.env.walls.lower()
    else:
        basedir = basedir / config.env.simulator_settings.scene.lower()

    if not basedir.exists():
        basedir.mkdir(parents=True)

    if args.collect_trajs:
        problem_setup = setup_problems(
            eval_env, agent, args, config, basedir, save=save
        )
    else:
        assert args.load_problem_set
        assert len(args.problem_set_file) > 0
        problem_setup = load_problem_set(
            args.problem_set_file, eval_env, agent, args.visual
        )

        if args.method_type == "unconstrained":
            if args.num_agents == 1:
                single_unconstrained_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
            else:
                multi_unconstrained_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
        elif args.method_type == "unconstrained_search":
            if args.num_agents == 1:
                single_unconstrained_search_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
            else:
                multi_unconstrained_search_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
        elif args.method_type == "unconstrained_search_ds":
            assert args.num_agents != 1
            multi_unconstrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                ds=True,
                save=save,
            )
        elif args.method_type == "constrained":
            if args.num_agents == 1:
                single_constrained_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
            else:
                multi_constrained_policy(
                    agent, eval_env, problem_setup, args, config, basedir, save=save
                )
        elif args.method_type == "constrained_search":
            if args.num_agents == 1:
                single_constrained_search_policy(
                    agent,
                    eval_env,
                    problem_setup,
                    args,
                    config,
                    trained_cost_limit,
                    basedir,
                    save=save,
                )
            else:
                multi_constrained_search_policy(
                    agent,
                    eval_env,
                    problem_setup,
                    args,
                    config,
                    trained_cost_limit,
                    basedir,
                    save=save,
                )
        elif args.method_type == "constrained_search_ds":
            assert args.num_agents != 1
            multi_constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                ds=True,
                save=save,
            )
        elif args.method_type == "risk_bounded_constrained_search_ds":
            assert args.num_agents != 1
            multi_constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                ds=True,
                risk_bounded=True,
                save=save,
            )
        else:
            raise ValueError("Invalid method type")


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        main()
    except Exception as e:
        print("Error: ", e)
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
