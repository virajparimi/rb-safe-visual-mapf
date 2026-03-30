import torch
import logging
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy


from pud.mapf.cbs import CBSSolver
from collect_safe_trajectory_records import (
    habitat_setup,
    pointenv_setup,
    load_problem_set,
    load_agent_and_env,
    TIMELIMIT,
    PROBLEM_INDEX,
    REPLAY_BUFFER_INDEX,
    UNCONSTRAINED_PDIST_INDEX,
)

MAX_TIMELIMIT = 120


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--visual", default=False, action="store_true")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument(
        "--traj_difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    args = parser.parse_args()
    return args


def try_problems(agent, eval_env, problem_setup, args, config, basedir):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    unconstrained_agent = deepcopy(agent)
    unconstrained_agent.load_state_dict(
        torch.load(
            args.unconstrained_ckpt_file, map_location="cuda:0", weights_only=True
        )
    )

    save_path = basedir / args.traj_difficulty / "problems"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    valid_pbs = []
    save_path = save_path / f"pbs_{args.num_agents}.npy"

    if save_path.exists():
        logging.info(f"Loading existing problems from {save_path}")
        valid_pbs = np.load(save_path, allow_pickle=True).tolist()
        logging.info(f"Loaded {len(valid_pbs)} problems")
        if len(valid_pbs) >= args.num_samples:
            logging.info("Already have enough problems, skipping generation.")
            return

    rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
    if habitat:
        _, rb_vec = rb_vec
    pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["step"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    problems = problem_setup[PROBLEM_INDEX].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    rb_graph = nx.Graph()
    pdist_combined = np.max(pdist, axis=0)
    pcost_combined = np.max(pcost, axis=0)

    max_search_steps = 7 if not habitat else 4

    for i, _ in enumerate(rb_vec):
        for j, _ in enumerate(rb_vec):
            cost = pcost_combined[i, j]
            length = pdist_combined[i, j]
            if length < max_search_steps:
                rb_graph.add_edge(
                    i, j, weight=float(length), step=1.0, cost=float(cost)
                )

    idx = -1
    if len(valid_pbs) > 0:
        idx = len(valid_pbs) - 1
        logging.info(f"Starting from problem index {idx + 1}")
    with tqdm(total=args.num_samples - len(valid_pbs)) as pbar:
        while len(valid_pbs) < args.num_samples:
            idx += 1
            skip_idx = False

            pb_graph = rb_graph.copy()

            pbs = problems[idx * args.num_agents : (idx + 1) * args.num_agents]  # noqa
            goals = [pb["goal"] for pb in pbs]
            starts = [pb["start"] for pb in pbs]

            # Ensure that each start, goal pair is unique
            tuple_starts = [tuple(start) for start in starts]
            if len(set(tuple_starts)) != args.num_agents:
                logging.debug(f"Duplicate starts for problem {idx}")
                continue
            tuple_goals = [tuple(goal) for goal in goals]
            if len(set(tuple_goals)) != args.num_agents:
                logging.debug(f"Duplicate goals for problem {idx}")
                continue

            normalized_goals = [eval_env.normalize_obs(goal) for goal in goals]
            normalized_starts = [eval_env.normalize_obs(start) for start in starts]

            # Ensure that the start and goal can be connected to the replay buffer
            goal_ids = []
            start_ids = []
            num_nodes = rb_vec.shape[0] - 1

            for agent_id in range(args.num_agents):

                start_ids.append(num_nodes + 1)
                goal_ids.append(num_nodes + 2)
                state = {
                    "goal": normalized_goals[agent_id],
                    "observation": normalized_starts[agent_id],
                }

                start_to_rb_dist = unconstrained_agent.get_pairwise_dist(
                    [state["observation"]],
                    rb_vec,
                    aggregate="min",
                    max_search_steps=max_search_steps,
                    masked=True,
                )
                rb_to_goal_dist = unconstrained_agent.get_pairwise_dist(
                    rb_vec,
                    [state["goal"]],
                    aggregate="min",
                    max_search_steps=max_search_steps,
                    masked=True,
                )
                start_to_rb_cost = agent.get_pairwise_cost(
                    [state["observation"]],
                    rb_vec,
                    aggregate="max",
                )
                rb_to_goal_cost = agent.get_pairwise_cost(
                    rb_vec,
                    [state["goal"]],
                    aggregate="max",
                )

                # Vectorized edge addition for speed
                start_to_rb_dist_flat = start_to_rb_dist.flatten()
                start_to_rb_cost_flat = start_to_rb_cost.flatten()
                rb_to_goal_dist_flat = rb_to_goal_dist.flatten()
                rb_to_goal_cost_flat = rb_to_goal_cost.flatten()
                rb_indices = np.arange(rb_vec.shape[0])

                # Edges from start to replay buffer nodes
                valid_start_mask = start_to_rb_dist_flat < max_search_steps
                for i in rb_indices[valid_start_mask]:
                    pb_graph.add_edge(
                        start_ids[agent_id],
                        i,
                        weight=float(start_to_rb_dist_flat[i]),
                        step=1.0,
                        cost=float(start_to_rb_cost_flat[i]),
                    )

                # Edges from replay buffer nodes to goal
                valid_goal_mask = rb_to_goal_dist_flat < max_search_steps
                for i in rb_indices[valid_goal_mask]:
                    pb_graph.add_edge(
                        i,
                        goal_ids[agent_id],
                        weight=float(rb_to_goal_dist_flat[i]),
                        step=1.0,
                        cost=float(rb_to_goal_cost_flat[i]),
                    )

                if not np.any(start_to_rb_dist < max_search_steps) or not np.any(
                    rb_to_goal_dist < max_search_steps
                ):
                    logging.debug(
                        f"Failed to connect start or goal to the replay buffer for problem {idx}"
                    )
                    skip_idx = True
                    break

                num_nodes += 2

            if skip_idx:
                continue

            # Ensure that vanilla CBS can find a solution for the problem

            augmented_wps = np.concatenate([rb_vec, normalized_starts, normalized_goals], axis=0)
            extended_pdist = unconstrained_agent.get_pairwise_dist(augmented_wps, aggregate=None)

            cbs_solver = CBSSolver(
                graph=pb_graph,
                pdist=extended_pdist,
                starts=start_ids,
                goals=goal_ids,
                config=cbs_config,
            )

            try:
                logging.debug(f"Finding path for problem {idx}")
                cbs_solver.find_paths()
            except Exception as e:
                logging.debug(f"Failed to find path for problem {idx} because {e}")
                continue

            valid_pb = {
                "graph": pb_graph,
                "starts": starts,
                "goals": goals,
                "pbs": pbs,
            }
            valid_pbs.append(valid_pb)
            np.save(save_path, valid_pbs)
            pbar.update(1)

    np.save(save_path, valid_pbs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

    problem_setup = load_problem_set(
        args.problem_set_file, args.visual
    )
    try_problems(agent, eval_env, problem_setup, args, config, basedir)
