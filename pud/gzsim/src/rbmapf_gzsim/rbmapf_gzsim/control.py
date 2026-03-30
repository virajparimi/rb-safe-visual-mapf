import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from copy import deepcopy


from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.envs.simple_navigation_env import PointEnv
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.envs.habitat_navigation_env import HabitatNavigationEnv, GoalConditionedHabitatPointWrapper
from pud.algos.policies import ConstrainedMultiAgentSearchPolicy, VisualConstrainedMultiAgentSearchPolicy
from pud.plots.collect_safe_trajectory_records import (
    TIMELIMIT,
    MAX_TIMELIMIT,
    PROBLEM_INDEX,
    REPLAY_BUFFER_INDEX,
    UNCONSTRAINED_PDIST_INDEX
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--visual", type=str, default="False")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument("--sdf_path", type=str, default="models/default.sdf")
    args, _ = parser.parse_known_args()
    return args


def pointenv_setup(args):
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    config.device = args.device
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

    return config, eval_env, agent


def habitat_setup(args):
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    config.device = args.device
    config.replay_buffer.max_size = args.replay_buffer_size
    config.env.simulator_settings.scene_dataset = "../../" + config.env.simulator_settings.scene_dataset

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

    config.agent["action_dim"] = action_dim  # type: ignore
    config.agent["max_action"] = max_action  # type: ignore

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

    return config, eval_env, agent


def denormalize(wp, height, width, z=2.0):
    print(f"Denormalizing: {wp}")
    ans = np.array([wp[0] * height, wp[1] * width], dtype=np.float32)
    print(f"Denormalized: {ans}")
    return np.array([wp[0] * height, wp[1] * width, z], dtype=np.float32)


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


def extract_walls(args):
    habitat = args.visual == "True"
    if habitat:
        _, eval_env, _ = habitat_setup(args)
    else:
        _, eval_env, _ = pointenv_setup(args)
    assert isinstance(eval_env.unwrapped, PointEnv) or isinstance(eval_env.unwrapped, HabitatNavigationEnv)

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    cols, rows = eval_env.get_map().shape
    normalizing_factor = np.array([cols, rows]) if habitat else np.ones(2)

    problem_setup = load_problem_set(args.problem_set_file, habitat)
    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    state, _ = eval_env.reset()  # type: ignore
    state, _ = eval_env.reset()  # type: ignore
    assert state is not None
    assert isinstance(state, dict)
    if habitat:
        agent_goal = [state["grid"]["goal"]]
        agent_start = [state["grid"]["observation"]]
    else:
        agent_goal = [state["goal"]]
        agent_start = [state["observation"]]

    goals = agent_goal.copy()
    starts = agent_start.copy()

    for _ in range(args.num_agents - 1):
        agent_state, _ = eval_env.reset()  # type: ignore
        assert agent_state is not None
        assert isinstance(agent_state, dict)
        if habitat:
            agent_goal = [agent_state["grid"]["goal"]]
            agent_start = [agent_state["grid"]["observation"]]
        else:
            agent_goal = [agent_state["goal"]]
            agent_start = [agent_state["observation"]]

        goals.extend(agent_goal.copy())
        starts.extend(agent_start.copy())

    denormed_starts = [
        denormalize(start / normalizing_factor, cols, rows) for start in starts]
    denormed_adjusted_starts = [
        adjust_positions(start, eval_env, habitat) for start in denormed_starts
    ]

    if habitat:
        assert isinstance(eval_env.unwrapped, HabitatNavigationEnv)
        return eval_env.get_map(), denormed_adjusted_starts, eval_env.unwrapped.get_bounds()
    else:
        return eval_env.get_map(), denormed_adjusted_starts


def adjust_positions(position, env, habitat):
    if habitat:
        lower_bounds, _ = env.unwrapped.get_bounds()
        habitat_x = lower_bounds[2] + position[0] * 0.4
        habitat_y = lower_bounds[0] + position[1] * 0.4
        position = np.array([habitat_y, -habitat_x, position[2]], dtype=np.float32)
    else:
        walls = env.get_map()
        rows, cols = walls.shape
        origin = np.array([-cols / 2.0, -rows / 2.0, 0.0])
        position += origin
    return position


def generate_wps(args, debug=False):

    habitat = args.visual == "True"
    if habitat:
        _, eval_env, agent = habitat_setup(args)
    else:
        _, eval_env, agent = pointenv_setup(args)

    problem_setup = load_problem_set(args.problem_set_file, habitat)

    assert isinstance(eval_env.unwrapped, PointEnv) or isinstance(eval_env.unwrapped, HabitatNavigationEnv)

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    rb = rb_vec if not habitat else rb_vec[1]
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,  # TODO: Make this change based on the drone size
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["cost"],
        "max_distance": eval_env.max_goal_dist,  # type: ignore
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    search_policy_cls = (
        ConstrainedMultiAgentSearchPolicy if not habitat else VisualConstrainedMultiAgentSearchPolicy
    )

    constrained_ma_search_policy = search_policy_cls(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        max_cost_limit=np.inf,
        max_search_steps=7 if not habitat else 4,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained = constrained_ma_search_policy.constraints is not None

    if constrained_ma_search_policy.open_loop:

        state = eval_env.reset()
        state = eval_env.reset()

        assert state is not None
        state, _ = state if constrained else (state, None)

        assert isinstance(state, dict)
        if habitat:
            agent_goal = [(state["grid"]["goal"], state["goal"])]
            agent_start = [(state["grid"]["observation"], state["observation"])]
        else:
            agent_goal = [state["goal"]]
            agent_start = [state["observation"]]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        for _ in range(args.num_agents - 1):

            agent_state = eval_env.reset()
            assert agent_state is not None
            agent_state, _ = agent_state if constrained else (agent_state, None)

            assert isinstance(agent_state, dict)
            if habitat:
                agent_goal = [(agent_state["grid"]["goal"], agent_state["goal"])]
                agent_start = [(agent_state["grid"]["observation"], agent_state["observation"])]
            else:
                agent_goal = [agent_state["goal"]]
                agent_start = [agent_state["observation"]]

            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].extend(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not be modified ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        constrained_ma_search_policy.select_action(state)
        waypoints = constrained_ma_search_policy.get_augmented_waypoints()

    wps = []
    cols, rows = eval_env.get_map().shape
    normalizing_factor = np.array([cols, rows])

    for agent_id in range(args.num_agents):

        agent_goal = goals[agent_id]
        if habitat:
            agent_goal, _ = agent_goal
            agent_goal = agent_goal / normalizing_factor

        agent_start = starts[agent_id]
        if habitat:
            agent_start, _ = agent_start
            agent_start = agent_start / normalizing_factor

        if habitat:
            waypoint_vec = (
                np.array([wp[0] for wp in waypoints[agent_id]]) / normalizing_factor
            )
        else:
            waypoint_vec = np.array(waypoints[agent_id])

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start}")
        print(f"Waypoints: {waypoint_vec}")
        print(f"Goal: {agent_goal}")
        print(f"Steps: {waypoint_vec.shape[0]}")
        print("-" * 10)

        waypoint_vec = np.array([agent_start, *waypoint_vec, agent_goal])
        denormed = [denormalize(wp, cols, rows) for wp in waypoint_vec]
        denormed_adjusted = [adjust_positions(wp, eval_env, habitat) for wp in denormed]
        wps.append(denormed_adjusted)

    if debug:
        if habitat:
            from pud.visualizers.visualize_habitat import visualize_compare_search, visualize_search_path
        else:
            from pud.visualizers.visualize import visualize_compare_search, visualize_search_path
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        eval_env.reset()
        figdir = Path("log")
        figdir.mkdir(parents=True, exist_ok=True)
        visualize_search_path(
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.num_agents,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_constrained_multi_agent_search.jpg").as_posix()
        )
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        eval_env.reset()
        visualize_compare_search(
            agent,
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.num_agents,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_compare_constrained_multi_agent.jpg").as_posix()
        )

    # Returned wps are denormalized and shifted to match the origin of simulation/hardware environment
    return wps
