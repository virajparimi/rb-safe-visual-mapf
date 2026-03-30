import time
import yaml
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from dotmap import DotMap

from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_global_seed, set_env_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_pointenv.pb_sampler import load_pb_set, sample_pbs_by_agent
from pud.algos.policies import (
    MultiAgentSearchPolicy,
    VisualMultiAgentSearchPolicy,
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

TIMELIMIT = 60
MAX_TIMELIMIT = 600
COST_LIMIT_FACTOR = 0.5
COLLISION_THRESHOLD = 1e-3

PCOST_INDEX = 3
PROBLEM_INDEX = 4
REPLAY_BUFFER_INDEX = 0
CONSTRAINED_PDIST_INDEX = 2
UNCONSTRAINED_PDIST_INDEX = 1


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
        torch.load(
            args.constrained_ckpt_file,
            map_location=torch.device(config.device),
            weights_only=True,
        )
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
                args.constrained_ckpt_file,
                map_location=torch.device(config.device),
                weights_only=True,
            )
        )
    else:
        agent.load_state_dict(
            torch.load(
                args.unconstrained_ckpt_file,
                map_location=torch.device(config.device),
                weights_only=True,
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
            args.unconstrained_ckpt_file, map_location=torch.device(config.device), weights_only=True
        )
    )
    with torch.no_grad():
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
        torch.load(
            args.constrained_ckpt_file, map_location=torch.device(config.device), weights_only=True
        )
    )
    with torch.no_grad():
        pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore
        constrained_pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore

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


def load_problem_set(file_path, habitat=False):
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
            (rb_vec_grid, rb_vec),
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
            "constrained",
            "unconstrained",
            "lagrangian_search",
            "biobjective_search",
            "collect_bounds_data",
            "risk_budgeted_search",
            "constrained_risk_search",
            "constrained_reward_search",
            "unconstrained_reward_search",
            "surplus_driven_uniform_search",
            "surplus_driven_utility_search",
            "full_constrained_risk_search",
            "full_constrained_reward_search",
            "tatonnement_driven_uniform_search",
            "tatonnement_driven_utility_search",
            "surplus_driven_inverse_utility_search",
            "tatonnement_driven_inverse_utility_search",
        ],
        default="unconstrained",
    )

    args, _ = parser.parse_known_args()
    return args


def unconstrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        unconstrained_records = np.load(save_path, allow_pickle=True)
        unconstrained_records = unconstrained_records.tolist()

    start_idx = len(unconstrained_records)
    logging.info(f"Starting from index: {start_idx}")

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                agent,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            unconstrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            unconstrained_records.append(records)

        if save:
            np.save(save_path, unconstrained_records)

    if save:
        np.save(save_path, unconstrained_records)
    return unconstrained_records


def constrained_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    constrained_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        constrained_records = np.load(save_path, allow_pickle=True)
        constrained_records = constrained_records.tolist()

    start_idx = len(constrained_records)
    logging.info(f"Starting from index: {start_idx}")

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                agent,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            constrained_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            constrained_records.append(records)

        if save:
            np.save(save_path, constrained_records)

    if save:
        np.save(save_path, constrained_records)
    return constrained_records


def unconstrained_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    basedir,
    save=False,
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=False
    )

    unconstrained_search_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        unconstrained_search_records = np.load(save_path, allow_pickle=True)
        unconstrained_search_records = unconstrained_search_records.tolist()

    start_idx = len(unconstrained_search_records)
    logging.info(f"Starting from index: {start_idx}")

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "edge_attributes": ["step"],
        "split_strategy": "disjoint",
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        MultiAgentSearchPolicy if not habitat else VisualMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        pdist=pdist,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        open_loop=True,
        cbs_config=cbs_config,
        no_waypoint_hopping=True,
        max_search_steps=7 if not habitat else 4,
    )

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        search_policy.planning_graph = processed_problems[pb_idx]["graph"]
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                search_policy,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            unconstrained_search_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            unconstrained_search_records.append(records)

        if save:
            np.save(save_path, unconstrained_search_records)

    if save:
        np.save(save_path, unconstrained_search_records)
    return unconstrained_search_records


def constrained_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    trained_cost_limit,
    basedir,
    save=False,
    full_risk=False,
    edge_attributes=["step"],
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    constrained_search_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if full_risk:
        bounds_data = []
        key = "lb" if "risk" in args.method_type else "ub"
        bounds_data_path = save_path / "risk_bounds"
        if not bounds_data_path.exists():
            bounds_data_path.mkdir(parents=True)
        bounds_data_path = bounds_data_path / f"{key}_{args.num_agents}.npy"
        if save and Path(bounds_data_path).exists():
            bounds_data = np.load(bounds_data_path, allow_pickle=True)
            bounds_data = bounds_data.tolist()

    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        constrained_search_records = np.load(save_path, allow_pickle=True)
        constrained_search_records = constrained_search_records.tolist()

    start_idx = len(constrained_search_records)
    logging.info(f"Starting from index: {start_idx}")

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": edge_attributes,
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        no_waypoint_hopping=True,
        max_search_steps=7 if not habitat else 4,
        max_cost_limit=(
            COST_LIMIT_FACTOR * trained_cost_limit if not full_risk else np.inf
        ),
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        if not full_risk:
            graph = processed_problems[pb_idx]["graph"]
            for u, v in graph.edges():
                if graph[u][v]["cost"] > trained_cost_limit * COST_LIMIT_FACTOR:
                    graph.remove_edge(u, v)
        else:
            graph = processed_problems[pb_idx]["graph"]
        search_policy.planning_graph = graph
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                search_policy,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            constrained_search_records.append(records)
            if full_risk:
                all_success = True
                bound_data = []
                for agent in range(args.num_agents):
                    bound_data.append(records[agent]["cumulative_costs"])
                    if not records[agent]["success"]:
                        all_success = False
                        break
                if all_success:
                    bounds_data.append(np.sum(bound_data))
                else:
                    bounds_data.append(-1)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            constrained_search_records.append(records)
            if full_risk:
                bounds_data.append(-1)

        if save:
            if full_risk:
                np.save(bounds_data_path, bounds_data)
            np.save(save_path, constrained_search_records)

    if save:
        if full_risk:
            np.save(bounds_data_path, bounds_data)
        np.save(save_path, constrained_search_records)
    return constrained_search_records


def lagrangian_search_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    lagrangian_search_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        lagrangian_search_records = np.load(save_path, allow_pickle=True)
        lagrangian_search_records = lagrangian_search_records.tolist()

    start_idx = len(lagrangian_search_records)
    logging.info(f"Starting from index: {start_idx}")

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["step", "cost"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "lagrangian": agent.lagrange.lagrangian_multiplier.data.numpy(),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        open_loop=True,
        pdist=pdist,
        pcost=pcost,
        cbs_config=cbs_config,
        no_waypoint_hopping=True,
        max_search_steps=7 if not habitat else 4,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        search_policy.planning_graph = processed_problems[pb_idx]["graph"]
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                search_policy,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            lagrangian_search_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            lagrangian_search_records.append(records)

        if save:
            np.save(save_path, lagrangian_search_records)

    if save:
        np.save(save_path, lagrangian_search_records)
    return lagrangian_search_records


def biobjective_search_policy(
    agent, eval_env, problem_setup, args, config, basedir, save=False
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    biobjective_search_records = []
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npy"
    if save and Path(save_path).exists():
        biobjective_search_records = np.load(save_path, allow_pickle=True)
        biobjective_search_records = biobjective_search_records.tolist()

    start_idx = len(biobjective_search_records)
    logging.info(f"Starting from index: {start_idx}")

    cbs_config = {
        "seed": None,
        "use_experience": False,
        "collision_radius": 0.0,
        "use_cardinality": False,
        "risk_attribute": "cost",
        "use_multi_objective": True,
        "split_strategy": "disjoint",
        "edge_attributes": ["step", "cost"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/bocbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        open_loop=True,
        pdist=pdist,
        pcost=pcost,
        cbs_config=cbs_config,
        no_waypoint_hopping=True,
        max_search_steps=7 if not habitat else 4,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for pb_idx in tqdm(range(start_idx, config.num_samples)):
        start_costs = np.zeros(args.num_agents)
        input_starts, input_goals = [], []
        input_starts = [
            (
                eval_env.normalize_obs(start)
                if not habitat
                else (start, eval_env.normalize_obs(start))
            )
            for start in processed_problems[pb_idx]["starts"]
        ]
        input_goals = [
            (
                eval_env.normalize_obs(goal)
                if not habitat
                else (goal, eval_env.normalize_obs(goal))
            )
            for goal in processed_problems[pb_idx]["goals"]
        ]
        search_policy.planning_graph = processed_problems[pb_idx]["graph"]
        try:
            start_time = time.time()
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                search_policy,
                eval_env,
                args.num_agents,
                habitat=habitat,
                input_starts=input_starts,
                input_goals=input_goals,
                start_costs=start_costs,
                wait=True,
                threshold=COLLISION_THRESHOLD,
            )
            records.append({"total_time": time.time() - start_time})
            biobjective_search_records.append(records)
        except Exception as e:
            logging.error(f"Error: {e}")
            records = [{} for _ in range(args.num_agents)]
            records.append({"total_time": time.time() - start_time})
            biobjective_search_records.append(records)

        if save:
            np.save(save_path, biobjective_search_records)

    if save:
        np.save(save_path, biobjective_search_records)
    return biobjective_search_records


def risk_budgeted_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    basedir,
    save=False,
):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    risk_percents = [0.0, 0.25, 0.5, 0.75, 1.0]
    risk_budgeted_search_records = [[] for _ in risk_percents]
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    cbs_bounds_data_path = save_path / "cbs_risk_bounds"
    cbs_lb_data = np.load(
        cbs_bounds_data_path / f"lb_{args.num_agents}.npy", allow_pickle=True
    ).tolist()
    cbs_ub_data = np.load(
        cbs_bounds_data_path / f"ub_{args.num_agents}.npy", allow_pickle=True
    ).tolist()

    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npz"
    if save and Path(save_path).exists():
        data = np.load(save_path, allow_pickle=True)
        for idx, pct in enumerate(risk_percents):
            if str(idx) in data.files:
                risk_budgeted_search_records[idx] = data[str(idx)].tolist()

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["step", "cost"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/rbcbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        max_search_steps=7 if not habitat else 4,
        max_cost_limit=np.inf,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for idx, pct in enumerate(risk_percents):

        start_idx = len(risk_budgeted_search_records[idx])
        logging.info(f"Starting from index: {start_idx}")

        problems = problem_setup[PROBLEM_INDEX].copy()
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        for pb_idx in tqdm(range(start_idx, config.num_samples)):
            lb = cbs_lb_data[pb_idx]
            ub = cbs_ub_data[pb_idx]
            if lb == -1 or ub == -1:
                record = [{} for _ in range(args.num_agents)]
                record.append({"total_time": 0.0})
                risk_budgeted_search_records[idx].append(record)
                continue
            elif lb == ub and pct != 0.0:
                risk_budgeted_search_records[idx].append(
                    risk_budgeted_search_records[idx - 1][pb_idx]
                )
                continue
            else:
                risk_budget = lb if lb == ub else lb + pct * (ub - lb)

            cbs_config["risk_budget"] = risk_budget
            start_costs = np.zeros(args.num_agents)
            input_starts, input_goals = [], []
            input_starts = [
                (
                    eval_env.normalize_obs(start)
                    if not habitat
                    else (start, eval_env.normalize_obs(start))
                )
                for start in processed_problems[pb_idx]["starts"]
            ]
            input_goals = [
                (
                    eval_env.normalize_obs(goal)
                    if not habitat
                    else (goal, eval_env.normalize_obs(goal))
                )
                for goal in processed_problems[pb_idx]["goals"]
            ]
            search_policy.planning_graph = processed_problems[pb_idx]["graph"]

            try:
                start_time = time.time()
                _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                    search_policy,
                    eval_env,
                    args.num_agents,
                    input_starts=input_starts,
                    input_goals=input_goals,
                    start_costs=start_costs,
                    habitat=habitat,
                    wait=True,
                    threshold=COLLISION_THRESHOLD,
                )
                records.append({"total_time": time.time() - start_time})
                risk_budgeted_search_records[idx].append(records)
            except Exception as e:
                logging.error(f"Error: {e}")
                records = [{} for _ in range(args.num_agents)]
                records.append({"total_time": time.time() - start_time})
                risk_budgeted_search_records[idx].append(records)

            if save:
                store_data = {str(idx): risk_budgeted_search_records[idx]}
                np.savez(save_path, **store_data)  # type: ignore

        if save:
            store_data = {str(idx): risk_budgeted_search_records[idx]}
            np.savez(save_path, **store_data)

    if save:
        data = {
            str(idx): risk_budgeted_search_records[idx]
            for idx in range(len(risk_percents))
        }
        np.savez(save_path, **data)  # type: ignore
        save_path = save_path.with_suffix(".npy")
        np.save(save_path, risk_budgeted_search_records)
    return risk_budgeted_search_records


def risk_bounded_search_policy(
    agent,
    eval_env,
    problem_setup,
    args,
    config,
    basedir,
    save=False,
    allocater="uniform",
    strategy="surplus_deficit",
):
    assert allocater in ["uniform", "utility", "inverse_utility"]

    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    risk_percents = [0.0, 0.25, 0.5, 0.75, 1.0]
    risk_bounded_search_records = [[] for _ in risk_percents]
    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    cbs_bounds_data_path = save_path / "cbs_risk_bounds"
    cbs_lb_data = np.load(
        cbs_bounds_data_path / f"lb_{args.num_agents}.npy", allow_pickle=True
    ).tolist()
    cbs_ub_data = np.load(
        cbs_bounds_data_path / f"ub_{args.num_agents}.npy", allow_pickle=True
    ).tolist()

    save_path = save_path / f"{args.method_type}_records_{args.num_agents}.npz"
    if save and Path(save_path).exists():
        data = np.load(save_path, allow_pickle=True)
        for idx, pct in enumerate(risk_percents):
            if str(idx) in data.files:
                risk_bounded_search_records[idx] = data[str(idx)].tolist()

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "budget_allocater": allocater,
        "edge_attributes": ["step", "cost"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "risk_reallocation_strategy": strategy,
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/rbcbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        max_search_steps=7 if not habitat else 4,
        max_cost_limit=np.inf,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for idx, pct in enumerate(risk_percents):

        start_idx = len(risk_bounded_search_records[idx])
        logging.info(f"Starting from index: {start_idx}")

        problems = problem_setup[PROBLEM_INDEX].copy()
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        for pb_idx in tqdm(range(start_idx, config.num_samples)):
            lb = cbs_lb_data[pb_idx]
            ub = cbs_ub_data[pb_idx]
            if lb == -1 or ub == -1:
                record = [{} for _ in range(args.num_agents)]
                record.append({"total_time": 0.0})
                risk_bounded_search_records[idx].append(record)
                continue
            elif lb == ub and pct != 0.0:
                risk_bounded_search_records[idx].append(
                    risk_bounded_search_records[idx - 1][pb_idx]
                )
                continue
            else:
                risk_budget = lb if lb == ub else lb + pct * (ub - lb)

            cbs_config["risk_bound"] = risk_budget
            start_costs = np.zeros(args.num_agents)
            input_starts, input_goals = [], []
            input_starts = [
                (
                    eval_env.normalize_obs(start)
                    if not habitat
                    else (start, eval_env.normalize_obs(start))
                )
                for start in processed_problems[pb_idx]["starts"]
            ]
            input_goals = [
                (
                    eval_env.normalize_obs(goal)
                    if not habitat
                    else (goal, eval_env.normalize_obs(goal))
                )
                for goal in processed_problems[pb_idx]["goals"]
            ]
            search_policy.planning_graph = processed_problems[pb_idx]["graph"]

            try:
                start_time = time.time()
                _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                    search_policy,
                    eval_env,
                    args.num_agents,
                    input_starts=input_starts,
                    input_goals=input_goals,
                    start_costs=start_costs,
                    habitat=habitat,
                    wait=True,
                    threshold=COLLISION_THRESHOLD,
                )
                records.append({"total_time": time.time() - start_time})
                risk_bounded_search_records[idx].append(records)
            except Exception as e:
                logging.error(f"Error: {e}")
                records = [{} for _ in range(args.num_agents)]
                records.append({"total_time": time.time() - start_time})
                risk_bounded_search_records[idx].append(records)

            if save:
                store_data = {str(idx): risk_bounded_search_records[idx]}
                np.savez(save_path, **store_data)  # type: ignore

        if save:
            store_data = {str(idx): risk_bounded_search_records[idx]}
            np.savez(save_path, **store_data)

    if save:
        data = {
            str(idx): risk_bounded_search_records[idx]
            for idx in range(len(risk_percents))
        }
        np.savez(save_path, **data)  # type: ignore
        save_path = save_path.with_suffix(".npy")
        np.save(save_path, risk_bounded_search_records)
    return risk_bounded_search_records


def collect_bounds_data(agent, eval_env, problem_setup, args, config, basedir):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    save_path = basedir / args.traj_difficulty
    processed_problems_path = save_path / "problems" / f"pbs_{args.num_agents}.npy"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    lb_bounds_data = []
    ub_bounds_data = []
    bounds_data_path = save_path / "cbs_risk_bounds"
    if not bounds_data_path.exists():
        bounds_data_path.mkdir(parents=True)
    lb_bounds_data_path = bounds_data_path / f"lb_{args.num_agents}.npy"
    ub_bounds_data_path = bounds_data_path / f"ub_{args.num_agents}.npy"
    if Path(lb_bounds_data_path).exists():
        lb_bounds_data = np.load(lb_bounds_data_path, allow_pickle=True)
        lb_bounds_data = lb_bounds_data.tolist()
    if Path(ub_bounds_data_path).exists():
        ub_bounds_data = np.load(ub_bounds_data_path, allow_pickle=True)
        ub_bounds_data = ub_bounds_data.tolist()

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "lb_save_path": lb_bounds_data_path,
        "ub_save_path": ub_bounds_data_path,
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT * 2),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    if args.num_agents >= 20:
        cbs_config["max_time"] = 600

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )
    search_policy = search_policy_cls(
        agent=agent,
        n_agents=args.num_agents,
        rb_vec=rb_vec,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        max_search_steps=7 if not habitat else 4,
        cbs_config=cbs_config,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )
    edge_attributes = [["step"], ["cost"]]

    processed_problems = np.load(processed_problems_path, allow_pickle=True)
    processed_problems = processed_problems.tolist()

    for edge_attrib in edge_attributes:

        if edge_attrib == ["cost"]:
            start_idx = len(lb_bounds_data)
        else:
            start_idx = len(ub_bounds_data)
        logging.info(f"Starting from index: {start_idx}")

        cbs_config["edge_attributes"] = edge_attrib

        problems = problem_setup[PROBLEM_INDEX].copy()
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        for pb_idx in tqdm(range(start_idx, config.num_samples)):
            start_costs = np.zeros(args.num_agents)
            input_starts, input_goals = [], []
            input_starts = [
                (
                    eval_env.normalize_obs(start)
                    if not habitat
                    else (start, eval_env.normalize_obs(start))
                )
                for start in processed_problems[pb_idx]["starts"]
            ]
            input_goals = [
                (
                    eval_env.normalize_obs(goal)
                    if not habitat
                    else (goal, eval_env.normalize_obs(goal))
                )
                for goal in processed_problems[pb_idx]["goals"]
            ]
            search_policy.planning_graph = processed_problems[pb_idx]["graph"]
            try:
                _, _, _, _, _, _ = ConstrainedCollector.get_trajectories(
                    search_policy,
                    eval_env,
                    args.num_agents,
                    habitat=habitat,
                    input_starts=input_starts,
                    input_goals=input_goals,
                    start_costs=start_costs,
                    wait=True,
                    threshold=COLLISION_THRESHOLD,
                )
            except Exception as e:
                logging.error(f"Error: {e}")
                if edge_attrib == ["cost"]:
                    lb_data = []
                    if Path(cbs_config["lb_save_path"]).exists():
                        lb_data = np.load(
                            cbs_config["lb_save_path"], allow_pickle=True
                        ).tolist()
                    lb_data.append(-1)
                    np.save(cbs_config["lb_save_path"], lb_data)
                else:
                    ub_data = []
                    if Path(cbs_config["ub_save_path"]).exists():
                        ub_data = np.load(
                            cbs_config["ub_save_path"], allow_pickle=True
                        ).tolist()
                    ub_data.append(-1)
                    np.save(cbs_config["ub_save_path"], ub_data)


def main():
    save = True
    args = argument_parser()
    if args.visual:
        config, eval_env, agent, trained_cost_limit = habitat_setup(args)
    else:
        config, eval_env, agent, trained_cost_limit = pointenv_setup(args)

    basedir = Path("pud/plots/data")
    if not args.visual:
        basedir = basedir / (config.env.walls.lower() + "_icaps")
    else:
        basedir = basedir / (config.env.simulator_settings.scene.lower() + "_icaps")

    if not basedir.exists():
        basedir.mkdir(parents=True)

    if args.collect_trajs:
        setup_problems(eval_env, agent, args, config, basedir, save=save)
    else:
        assert args.load_problem_set
        assert len(args.problem_set_file) > 0
        problem_setup = load_problem_set(args.problem_set_file, args.visual)

        if args.method_type == "unconstrained":
            unconstrained_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "unconstrained_reward_search":
            unconstrained_search_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "constrained":
            constrained_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "constrained_reward_search":
            constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                save=save,
            )
        elif args.method_type == "constrained_risk_search":
            constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                save=save,
                edge_attributes=["cost"],
            )
        elif args.method_type == "full_constrained_reward_search":
            constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                save=save,
                full_risk=True,
                edge_attributes=["step"],
            )
        elif args.method_type == "full_constrained_risk_search":
            constrained_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                trained_cost_limit,
                basedir,
                save=save,
                full_risk=True,
                edge_attributes=["cost"],
            )
        elif args.method_type == "lagrangian_search":
            lagrangian_search_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "biobjective_search":
            biobjective_search_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "risk_budgeted_search":
            risk_budgeted_search_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "surplus_driven_uniform_search":
            risk_bounded_search_policy(
                agent, eval_env, problem_setup, args, config, basedir, save=save
            )
        elif args.method_type == "surplus_driven_utility_search":
            risk_bounded_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                save=save,
                allocater="utility",
            )
        elif args.method_type == "surplus_driven_inverse_utility_search":
            risk_bounded_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                save=save,
                allocater="inverse_utility",
            )
        elif args.method_type == "tatonnement_driven_uniform_search":
            risk_bounded_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                save=save,
                strategy="price_clearing",
            )
        elif args.method_type == "tatonnement_driven_utility_search":
            risk_bounded_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                save=save,
                allocater="utility",
                strategy="price_clearing",
            )
        elif args.method_type == "tatonnement_driven_inverse_utility_search":
            risk_bounded_search_policy(
                agent,
                eval_env,
                problem_setup,
                args,
                config,
                basedir,
                save=save,
                allocater="inverse_utility",
                strategy="price_clearing",
            )
        elif args.method_type == "collect_bounds_data":
            collect_bounds_data(agent, eval_env, problem_setup, args, config, basedir)
        else:
            raise ValueError("Invalid method type")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    # try:
    #     logging.basicConfig(level=logging.INFO)
    #     main()
    # except Exception as e:
    #     print("Error: ", e)
    #     traceback.print_exc()
    #     sys.exit(1)
    # sys.exit(0)
