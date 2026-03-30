import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pud.collectors.collector import Collector
from pud.utils import set_env_seed, set_global_seed
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.safe_wrappers import set_safe_env_difficulty
from pud.envs.simple_navigation_env import plot_walls, set_env_difficulty
from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls, plot_trajs

AGENT_COLORS = [
    ("darkblue", "#1f77b4"),
    ("darkmagenta", "#d62728"),
    ("darkgreen", "#2ca02c"),
    ("darkcyan", "#9467bd"),
    ("darkred", "red"),
    ("darkorange", "orange"),
    ("black", "gray"),
]


def wall_plotting_fn(eval_env, ax, constrained=False):
    if not constrained:
        ax = plot_walls(eval_env.walls, ax)
    else:
        ax = plot_safe_walls(
            eval_env.walls, eval_env.get_cost_map(), eval_env.cost_limit, ax
        )
    return ax


def plot_agent_paths(
    a_id, start, goal, obs, title, ax, wps=None, obs_marker="o", use_agent_id=True
):

    end_label = "End " + str(a_id) if use_agent_id else "End"
    goal_label = "Goal " + str(a_id) if use_agent_id else "Goal"
    start_label = "Start " + str(a_id) if use_agent_id else "Start"
    waypoint_label = "Waypoint " + str(a_id) if use_agent_id else "Waypoint"

    ax.plot(obs[:, 0], obs[:, 1], obs_marker + "-", c=AGENT_COLORS[a_id][1], alpha=0.8)
    ax.scatter(
        [start[0]],
        [start[1]],
        marker="+",
        c=AGENT_COLORS[a_id][1],
        s=200,
        label=start_label,
    )
    ax.scatter(
        [obs[-1, 0]],
        [obs[-1, 1]],
        marker="x",
        c=AGENT_COLORS[a_id][1],
        s=200,
        label=end_label,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        c=AGENT_COLORS[a_id][1],
        s=200,
        label=goal_label,
    )
    if wps is not None:
        ax.plot(
            [start[0], *wps[:, 0]],
            [start[1], *wps[:, 1]],
            "s-",
            c=AGENT_COLORS[a_id][1],
            alpha=0.3,
            label=waypoint_label,
        )
    ax.set_title(title, fontsize=16)
    return ax


def visualize_trajectory(agent, eval_env, difficulty=0.5, outpath=""):

    constrained = hasattr(agent, "constraints") and agent.constraints is not None

    if constrained:
        cost_constraints = agent.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(8, 4))
    for trajectory in range(2):

        ax = plt.subplot(1, 2, trajectory + 1)
        ax = wall_plotting_fn(eval_env, ax, constrained)

        collector_cls = ConstrainedCollector if constrained else Collector
        start, goal, observations, _, _, records = collector_cls.get_trajectory(
            agent, eval_env
        )

        obs_vec = np.array(observations)
        print(f"Trajectory {trajectory}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Reward: {records['rewards']}")
        print(f"Steps: {records['steps']}")
        if "max_step_cost" in records:
            print(f"Max Step Cost: {records['max_step_cost']}")
            print(f"Cumulative Cost: {records['cumulative_costs']}")
        print("-" * 10)

        ax = plot_agent_paths(
            0, start, goal, obs_vec, "Trajectory " + str(trajectory + 1), ax
        )

    plt.legend(loc="lower center", bbox_to_anchor=(-0.1, -0.2), ncol=4, fontsize=16)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_buffer(rb_vec, eval_env, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_walls(eval_env.walls, ax=ax)
    ax.scatter(rb_vec[:, 0], rb_vec[:, 1])
    plt.title(f"Replay Buffer (Size: {rb_vec.shape[0]})", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_pairwise_dists(pdist, outpath=""):
    plt.figure(figsize=(7, 4))
    plt.hist(pdist.flatten(), bins=range(20))
    plt.xlabel("Predicted Distance")
    plt.ylabel("Number of (s, g) pairs")
    plt.title("Pairwise Distance Distribution")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_pairwise_costs(pdist, cost_limit, n_bins=20, outpath=""):
    plt.figure(figsize=(7, 4))
    plt.hist(pdist.flatten(), bins=np.linspace(0, cost_limit, n_bins))
    plt.xlabel("Predicted Costs")
    plt.ylabel("Number of (s, g) pairs")
    plt.title("Pairwise Cost Distribution")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_eval_records(
    eval_records, eval_env, ax, starts=[], goals=[], use_pbar=False, color=None
):

    list_trajs = []
    for id in eval_records.keys():
        list_trajs.append(eval_records[id]["traj"])

    ax = plot_safe_walls(
        walls=eval_env.get_map(),
        cost_map=eval_env.get_cost_map(),
        cost_limit=eval_env.cost_limit,
        ax=ax,
    )

    ax = plot_trajs(
        ax=ax,
        s=32,
        goals=goals,
        starts=starts,
        traj_color=color,
        use_pbar=use_pbar,
        list_trajs=list_trajs,
        walls=eval_env.get_map(),
    )

    return ax


def visualize_problems(eval_env, ax, starts=[], goals=[]):
    return visualize_eval_records(
        eval_records={}, eval_env=eval_env, ax=ax, starts=starts, goals=goals
    )


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    plot_walls(eval_env.walls, ax)
    ax.scatter(*rb_vec.T)

    pdist_combined = np.max(pdist, axis=0)
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.title("Graph", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_cost_graph(
    rb_vec, eval_env, pcost, cost_limit, outpath="", edges_to_display=8
):
    # Plot the edges that are deemed unsafe
    pcost_combined = np.max(pcost, axis=0)  # rb_vec, rb_vec
    safe_mask = pcost_combined < cost_limit
    ind_v, _ = np.where(safe_mask)
    print(
        "Ratio of predicted unsafe edges: {:.2f}%".format(
            100.0 * len(ind_v) / np.prod(safe_mask.shape)
        )
    )
    assert len(ind_v) == len(ind_v)

    _, ax = plt.subplots()
    plot_safe_walls(
        eval_env.get_map(), eval_env.get_cost_map(), cost_limit=cost_limit, ax=ax
    )
    ax.scatter(rb_vec[:, 0], rb_vec[:, 1])

    pbar = tqdm(total=len(rb_vec))
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pcost_combined[i])):
            if count < edges_to_display and pcost_combined[i, j] < cost_limit:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="g", alpha=0.5)
        pbar.update()

    plt.title(f"Cost Graph (Cost Limit: {cost_limit})")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_combined_graph(
    rb_vec, eval_env, pdist, pcost, cost_limit, cutoff=7, outpath="", edges_to_display=8
):
    """
    Plot edges that are both within the cutoff distance and cost limit shorter edges are prioritized
    rb_vec, pdist, pcost: (ensemble_size, N, N)
    """

    _, ax = plt.subplots()
    ax.scatter(*rb_vec.T)
    ax = plot_safe_walls(
        eval_env.get_map(), eval_env.get_cost_map(), cost_limit=cost_limit, ax=ax
    )

    pbar = tqdm(total=len(rb_vec))
    pdist_combined = np.max(pdist, axis=0)
    pcost_combined = np.max(pcost, axis=0)  # rb_vec, rb_vec
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if (
                count < edges_to_display
                and pdist_combined[i, j] < cutoff
                and pcost_combined[i, j] < cost_limit
            ):
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="g", alpha=0.4)
        pbar.update()

    plt.title(f"Combined Graph (Cost Limit: {cost_limit})")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_policy_graph(search_policy, eval_env, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*search_policy.rb_vec.T)
    plot_safe_walls(
        eval_env.get_map(),
        eval_env.get_cost_map(),
        cost_limit=eval_env.cost_limit,
        ax=ax,
    )

    edges_to_display = 8
    constrained = True if "constrain" in type(search_policy).__name__.lower() else False

    pdist_combined = np.max(search_policy.pdist, axis=0)
    pcost_combined = np.max(search_policy.pcost, axis=0) if constrained else None
    for i, s_i in enumerate(search_policy.rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display:
                if pdist_combined[i, j] < search_policy.max_search_steps:
                    if (
                        pcost_combined is not None
                        and pcost_combined[i, j] >= search_policy.max_cost_limit
                    ):
                        continue
                    s_j = search_policy.rb_vec[j]
                    ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph_ensemble(
    rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""
):
    ensemble_size = pdist.shape[0]
    _, ax = plt.subplots(nrows=1, ncols=ensemble_size, figsize=(5 * ensemble_size, 6))

    for col_index in range(ensemble_size):
        ax[col_index] = plot_walls(eval_env.walls, ax=ax[col_index])  # type: ignore
        ax[col_index].set_title("Critic %d" % (col_index + 1))  # type: ignore
        ax[col_index].scatter(*rb_vec.T)  # type: ignore
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    ax[col_index].plot(  # type: ignore
                        [s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5
                    )

    plt.suptitle("Graph Ensemble", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_combined_graph_ensemble(
    rb_vec, eval_env, pdist, pcost, cost_limit, cutoff=7, edges_to_display=8, outpath=""
):

    ensemble_size = pdist.shape[0]
    _, ax = plt.subplots(nrows=1, ncols=ensemble_size, figsize=(5 * ensemble_size, 5))

    for col_index in range(ensemble_size):
        ax[col_index] = plot_safe_walls(  # type: ignore
            eval_env.get_map(),
            eval_env.get_cost_map(),
            cost_limit=cost_limit,
            ax=ax[col_index],  # type: ignore
        )
        ax[col_index].set_title("Critic %d" % (col_index + 1))  # type: ignore
        ax[col_index].scatter(*rb_vec.T)  # type: ignore
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if (
                    count < edges_to_display
                    and pdist[col_index, i, j] < cutoff
                    and pcost[col_index, i, j] < cost_limit
                ):
                    s_j = rb_vec[j]
                    ax[col_index].plot(  # type: ignore
                        [s_i[0], s_j[0]], [s_i[1], s_j[1]], c="g", alpha=0.4
                    )

    plt.suptitle("Combined Graph Ensemble", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_full_graph(g, rb_vec, eval_env, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_walls(eval_env.walls, ax)
    ax.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.title(f"|V|={g.number_of_nodes()}, |E|={len(edges_to_plot)}")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_path(paths, filename, plot_handles, save_fig=False, waypoints=None):

    num_agents = len(paths)
    fig, ax = plot_handles

    ends = []
    lines = []
    starts = []
    waypoint_lines = []

    for agent in range(num_agents):
        (line,) = ax.plot(
            [],
            [],
            "o-",
            ms=10,
            lw=2,
            c=AGENT_COLORS[agent][1],
            alpha=0.7,
            label=f"Agent {agent}",
        )
        (start,) = ax.plot(
            [],
            [],
            "x",
            ms=10,
            lw=2,
            c=AGENT_COLORS[agent][0],
            alpha=0.7,
            label=f"Start {agent}",
        )
        (end,) = ax.plot(
            [],
            [],
            "*",
            ms=10,
            lw=2,
            c=AGENT_COLORS[agent][0],
            alpha=0.7,
            label=f"Goal {agent}",
        )
        if waypoints is not None:
            (waypoint_line,) = ax.plot(
                [],
                [],
                "s-",
                ms=10,
                lw=2,
                c=AGENT_COLORS[agent][1],
                alpha=0.7,
                label=f"Waypoint {agent}",
            )
            waypoint_lines.append(waypoint_line)
        ends.append(end)
        lines.append(line)
        starts.append(start)

    def init():
        for i, path in enumerate(paths):
            starts[i].set_data([path[0][0]], [path[0][1]])
            ends[i].set_data([path[-1][0]], [path[-1][1]])
        return starts

    def update(frame):
        for i, path in enumerate(paths):
            if frame >= len(path):
                continue
            x_data = [point[0] for point in path[: frame + 1]]
            y_data = [point[1] for point in path[: frame + 1]]
            lines[i].set_data(x_data, y_data)
            if waypoints is not None:
                wps_x_data = [point[0] for point in waypoints[i][: frame + 1]]
                wps_y_data = [point[1] for point in waypoints[i][: frame + 1]]
                waypoint_lines[i].set_data(wps_x_data, wps_y_data)

        return lines if waypoints is None else lines + waypoint_lines

    frames = max(len(path) for path in paths)
    plt.title("Agent Paths Visualization", fontsize=24)
    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=num_agents, fontsize=16
    )
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, repeat=False
    )
    if save_fig:
        ani.save(filename, writer="pillow", fps=1)


def visualize_search_path_single_agent(
    search_policy, eval_env, difficulty=0.5, outpath=""
):

    constrained = search_policy.constraints is not None

    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:
        state = eval_env.reset()
        state, _ = state if constrained else (state, None)

        goal = state["goal"]
        start = state["observation"]

        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        start, goal, _, waypoints, _, _ = collector_cls.get_trajectory(
            search_policy, eval_env
        )

    _, ax = plt.subplots(figsize=(6, 6))
    ax = wall_plotting_fn(eval_env, ax, constrained)

    waypoint_vec = np.array(waypoints)

    print(f"Start: {start}")
    print(f"Waypoints: {waypoint_vec}")
    print(f"Goal: {goal}")
    print(f"Steps: {waypoint_vec.shape[0]}")
    print("-" * 10)

    ax = plot_agent_paths(
        0,
        start,
        goal,
        np.array([start, *waypoint_vec, goal]),
        "Search",
        ax,
        obs_marker="s",
    )

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_search_path_multi_agent(
    search_policy, eval_env, num_agents, difficulty=0.5, outpath=""
):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:

        state = eval_env.reset()
        state, _ = state if constrained else (state, None)

        agent_goal = [state["goal"]]
        agent_start = [state["observation"]]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        for _ in range(num_agents - 1):

            agent_state = eval_env.reset()
            agent_state, _ = agent_state if constrained else (agent_state, None)

            agent_goal = [agent_state["goal"]]
            agent_start = [agent_state["observation"]]

            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].append(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not be modified ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        search_policy.select_action(state)
        waypoints = search_policy.get_augmented_waypoints()
    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        starts, goals, _, waypoints, _, _ = collector_cls.get_trajectories(
            search_policy, eval_env, num_agents
        )

    _, ax = plt.subplots(figsize=(8, 9))
    ax = wall_plotting_fn(eval_env, ax, constrained)

    agent_waypoints = []
    for agent_id in range(num_agents):

        agent_goal = goals[agent_id]
        agent_start = starts[agent_id]

        waypoint_vec = np.array(waypoints[agent_id])

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start}")
        print(f"Waypoints: {waypoint_vec}")
        print(f"Goal: {agent_goal}")
        print(f"Steps: {waypoint_vec.shape[0]}")
        print("-" * 10)

        waypoint_vec = np.array([agent_start, *waypoint_vec, agent_goal])
        agent_waypoints.append(waypoint_vec)

        ax = plot_agent_paths(
            agent_id,
            agent_start,
            agent_goal,
            waypoint_vec,
            "Search",
            ax,
            obs_marker="s",
        )

    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=num_agents, fontsize=16
    )
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        save_path = outpath[:-4] + ".gif"
        fig, ax = plt.subplots(figsize=(8, 9))
        ax = wall_plotting_fn(eval_env, ax, constrained)
        visualize_path(agent_waypoints, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_search_path(
    search_policy, eval_env, difficulty=0.5, outpath="", num_agents=None
):
    if num_agents is None:
        visualize_search_path_single_agent(search_policy, eval_env, difficulty, outpath)
    else:
        visualize_search_path_multi_agent(
            search_policy, eval_env, num_agents, difficulty, outpath
        )


def visualize_compare_search_single_agent(
    agent, search_policy, eval_env, difficulty=0.5, seed=0, outpath=""
):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(12, 6))

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = wall_plotting_fn(eval_env, ax, constrained)

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent

        collector_cls = ConstrainedCollector if constrained else Collector
        if col_index == 0:
            start, goal, observations, waypoints, _, records = (
                collector_cls.get_trajectory(policy, eval_env)
            )
        else:
            start_cost = (
                records["first_step_cost"] if "first_step_cost" in records else None
            )
            _, _, observations, waypoints, _, records = collector_cls.get_trajectory(
                policy, eval_env, start, goal, start_cost
            )

        obs_vec = np.array(observations)
        waypoint_vec = np.array([start, *waypoints, goal]) if use_search else None

        print(f"Policy: {title}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Reward: {records['rewards']}")
        print(f"Steps: {records['steps']}")
        if "max_step_cost" in records:
            print(f"Max Step Cost: {records['max_step_cost']}")
            print(f"Cumulative Cost: {records['cumulative_costs']}")
        print("-" * 10)

        ax = plot_agent_paths(0, start, goal, obs_vec, title, ax, waypoint_vec)
        if not use_search:
            ax.legend(
                loc="lower center", bbox_to_anchor=(-0.15, -0.15), ncol=4, fontsize=16
            )

    plt.suptitle("Single-Agent Search Comparison", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_compare_search_multi_agent(
    agent, search_policy, eval_env, n_agents, difficulty=0.5, seed=0, outpath=""
):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(10, 6))

    search_waypoints = []
    search_observations = []
    no_search_observations = []

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = wall_plotting_fn(eval_env, ax, constrained)

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        threshold = 0.0
        policy = search_policy if use_search else agent
        collector_cls = ConstrainedCollector if constrained else Collector

        if col_index == 0:
            starts, goals, observations, waypoints, _, records = (
                collector_cls.get_trajectories(
                    policy, eval_env, n_agents, threshold=threshold
                )
            )
        else:
            start_costs = []
            for a_id in range(n_agents):
                if "first_step_cost" in records[a_id]:
                    start_costs.append(records[a_id]["first_step_cost"])
                else:
                    start_costs = None
                    break
            _, _, observations, waypoints, _, records = collector_cls.get_trajectories(
                policy,
                eval_env,
                n_agents,
                starts,
                goals,
                start_costs=start_costs,
                threshold=threshold,
            )

        print(f"Policy: {title}")
        for a_id in range(n_agents):

            a_goal = goals[a_id]
            a_start = observations[a_id][0]

            obs_vec = np.array(observations[a_id])
            waypoint_vec = (
                np.array([a_start, *waypoints[a_id], a_goal]) if use_search else None
            )

            print(f"Agent: {a_id}")
            print(f"Start: {a_start}")
            if use_search:
                search_waypoints.append(waypoint_vec)
                search_observations.append(obs_vec)
                print(f"Waypoints: {waypoint_vec}")
            else:
                no_search_observations.append(obs_vec)
            print(f"Goal: {a_goal}")
            print(f"Reward: {records[a_id]['rewards']}")
            print(f"Steps: {records[a_id]['steps']}")
            if "max_step_cost" in records[a_id]:
                print(f"Max Step Cost: {records[a_id]['max_step_cost']}")
                print(f"Cumulative Cost: {records[a_id]['cumulative_costs']}")
            print("-" * 10)

            ax = plot_agent_paths(
                a_id, a_start, a_goal, obs_vec, title, ax, waypoint_vec
            )

        if not use_search:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(-0.15, -0.35),
                ncol=n_agents,
                fontsize=16,
            )

    plt.suptitle("Multi-Agent Search Comparison", fontsize=24)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)

        save_path = outpath[:-4] + "_search.gif"
        fig, ax = plt.subplots(figsize=(10, 11))
        ax = wall_plotting_fn(eval_env, ax, constrained)
        visualize_path(
            search_observations,
            save_path,
            (fig, ax),
            save_fig=True,
            waypoints=search_waypoints,
        )

        save_path = outpath[:-4] + "_no_search.gif"
        fig, ax = plt.subplots(figsize=(10, 11))
        ax = wall_plotting_fn(eval_env, ax, constrained)
        visualize_path(no_search_observations, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_compare_search(
    agent, search_policy, eval_env, difficulty=0.5, seed=0, outpath="", num_agents=None
):
    if num_agents is None:
        visualize_compare_search_single_agent(
            agent, search_policy, eval_env, difficulty, seed, outpath
        )
    else:
        visualize_compare_search_multi_agent(
            agent, search_policy, eval_env, num_agents, difficulty, seed, outpath
        )
