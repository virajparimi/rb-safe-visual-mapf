import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from matplotlib.animation import FuncAnimation

from pud.collectors.collector import Collector
from pud.utils import set_env_seed, set_global_seed
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.visualizers.visualize import plot_agent_paths, visualize_path
from pud.envs.habitat_navigation_env import plot_wall, set_habitat_env_difficulty

USE_GIFS = True
COLLISION_THRESHOLD = 1e-3
extension = ".gif" if USE_GIFS else ".mp4"


def visualize_trajectory(agent, eval_env, difficulty=0.5, outpath=""):

    constrained = hasattr(agent, "constraints") and agent.constraints is not None
    set_habitat_env_difficulty(eval_env.env, difficulty)

    height, width = eval_env.walls.shape
    normalizing_factor = np.array([height, width])

    plt.figure(figsize=(8, 4))
    for trajectory in range(2):

        ax = plt.subplot(1, 2, trajectory + 1)
        ax = plot_wall(eval_env.walls, ax)

        collector_cls = ConstrainedCollector if constrained else Collector
        start, goal, observations, _, _, records = collector_cls.get_trajectory(
            agent, eval_env, habitat=True
        )

        goal_grid, _ = goal
        goal_grid = goal_grid / normalizing_factor

        start_grid, start_visual = start
        start_grid = start_grid / normalizing_factor

        observations_grid = (
            np.array([obs[0] for obs in observations]) / normalizing_factor
        )
        obs_vec = np.array(observations_grid)

        print(f"Trajectory {trajectory}")
        print(f"Start: {start_grid}")
        print(f"Goal: {goal_grid}")
        print(f"Reward: {records['rewards']}")
        print(f"Steps: {records['steps']}")
        if "max_step_cost" in records:
            print(f"Max Step Cost: {records['max_step_cost']}")
            print(f"Cumulative Cost: {records['cumulative_costs']}")
        print("-" * 10)

        ax = plot_agent_paths(
            0, start_grid, goal_grid, obs_vec, "Trajectory " + str(trajectory + 1), ax
        )

    plt.legend(loc="lower center", bbox_to_anchor=(-0.1, -0.2), ncol=4, fontsize=16)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_buffer(rb_vec, eval_env, outpath=""):
    _, ax = plt.subplots()
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    scaled_rb_vec = rb_vec / np.array([height, width])
    ax.scatter(*scaled_rb_vec.T)
    plt.title(f"Replay Buffer (Size: {rb_vec.shape[0]})", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    scaled_rb_vec = rb_vec / np.array([height, width])
    ax.scatter(*scaled_rb_vec.T)

    pdist_combined = np.max(pdist, axis=0)
    for i, s_i in enumerate(scaled_rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = scaled_rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.title("Graph", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph_ensemble(
    rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""
):
    ensemble_size = pdist.shape[0]
    plt.figure(figsize=(5 * ensemble_size, 6))
    height, width = eval_env.walls.shape
    rb_vec = rb_vec / np.array([height, width])

    for col_index in range(ensemble_size):

        ax = plt.subplot(1, ensemble_size, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        plt.title("Critic %d" % (col_index + 1))
        plt.scatter(*rb_vec.T)

        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.suptitle("Graph Ensemble", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def as_mp4(frames, outpath, bs=1, border_color=(1, 0, 0, 0), fps=10):
    N, H, W, C = frames[0].shape
    final_height = H + 2 * bs
    final_width = W * N + bs * (N + 1)

    final_frames = []
    for frame in frames:

        ff = np.zeros((final_height, final_width, C), dtype=frame.dtype)
        for i in range(C):
            ff[:, :, i] = border_color[i]
        for i in range(N):
            start_x = bs + i * (W + bs)
            ff[bs : H + bs, start_x : start_x + W, :] = frame[i]  # noqa
            final_frames.append(ff)

    clip = ImageSequenceClip(final_frames[:-1], fps=10)
    clip.write_videofile(outpath, fps=10)

    goal_image = Image.fromarray(final_frames[-1])
    goal_image.save(outpath[:-4] + "_goal.png")


def as_gif(frames, outpath):

    image_arrays = []
    titles = ["Forward", "Right", "Backward", "Left"]

    for _, frame in enumerate(frames):
        image_arr = []
        for image in frame:
            rgb_img = Image.fromarray(image, mode="RGBA")
            image_arr.append(rgb_img)
        image_arrays.append(image_arr)

    fig = plt.figure(figsize=(10, 4))

    image_counter = 1
    image_arr = []
    for image_direction in range(len(titles)):
        ax = plt.subplot(1, 4, image_direction + 1)
        ax.axis("off")
        ax.set_title(titles[image_direction])
        ax_im = ax.imshow(image_arrays[0][image_direction])
        image_arr.append(ax_im)
    step_text = plt.text(-50.0, 40.0, "Step 0", fontsize=16)

    def update(*args):
        nonlocal image_counter
        if image_counter >= len(image_arrays) - 2:  # type: ignore
            image_counter = 0  # type: ignore
        else:
            image_counter += 1
        for j, image in enumerate(image_arrays[image_counter]):
            image_arr[j].set_array(image)
        step_text.set_text(f"Step {image_counter}")
        return image_arr

    plt.suptitle("Agent Paths Visualization", fontsize=24)
    ani = FuncAnimation(
        fig, update, fargs=(image_counter,), frames=len(image_arrays), blit=True
    )
    ani.save(outpath, writer="pillow", fps=1)

    N, H, W, C = frames[0].shape
    final_height = H + 2 * 1
    final_width = W * N + 1 * (N + 1)
    goal_frame = np.zeros((final_height, final_width, C), dtype=frame.dtype)
    for i in range(C):
        goal_frame[:, :, i] = 1 if i == 0 else 0
    for i in range(N):
        start_x = 1 + i * (W + 1)
        goal_frame[1 : H + 1, start_x : start_x + W, :] = image_arrays[-1][i]  # noqa

    goal_image = Image.fromarray(goal_frame)
    goal_image.save(outpath[:-4] + "_goal.png")


def visualize_habitat_agent(frames, outpath):

    if outpath[-4:] == ".mp4":
        as_mp4(frames, outpath)
    else:
        as_gif(frames, outpath)


def visualize_search_path_single_agent(
    search_policy, eval_env, outpath="", difficulty=0.5
):

    constrained = search_policy.constraints is not None
    safe = "safe" in type(eval_env.env).__name__.lower()  # type: ignore
    set_habitat_env_difficulty(eval_env.env, difficulty)

    if search_policy.open_loop:

        state = eval_env.reset()
        state = state[0] if safe else state
        goal = (state["grid"]["goal"], state["goal"])
        start = (state["grid"]["observation"], state["observation"])
        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        start, goal, _, waypoints, _, _ = collector_cls.get_trajectory(
            search_policy, eval_env, habitat=True
        )

    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    normalizing_factor = np.array([height, width])

    goal_grid, goal_visual = goal
    goal_grid = goal_grid / normalizing_factor

    start_grid, start_visual = start
    start_grid = start_grid / normalizing_factor

    waypoints_grid = np.array([wp[0] for wp in waypoints]) / normalizing_factor
    waypoints_visual = [wp[1] for wp in waypoints]

    print(f"Start: {start_grid}")
    print(f"Waypoints: {waypoints_grid}")
    print(f"Goal: {goal_grid}")
    print(f"Steps: {waypoints_grid.shape[0] - 1}")
    print("-" * 10)

    ax = plot_agent_paths(
        0,
        start_grid,
        goal_grid,
        np.array([start_grid, *waypoints_grid, goal_grid]),
        "Search",
        ax,
        obs_marker="s",
    )

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        visualize_habitat_agent(
            [start_visual, *waypoints_visual, goal_visual], outpath[:-4] + extension
        )
    else:
        plt.show()


def visualize_search_path_multi_agent(
    search_policy, eval_env, num_agents, outpath="", difficulty=0.5
):

    constrained = search_policy.constraints is not None
    set_habitat_env_difficulty(eval_env.env, difficulty)

    if search_policy.open_loop:

        state, _ = eval_env.reset()
        # Use the sampled start and goal for the first agent
        agent_goal = [(state["grid"]["goal"], state["goal"])]
        agent_start = [(state["grid"]["observation"], state["observation"])]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        # Sample the starts and goals for the other agents
        for _ in range(num_agents - 1):

            agent_state, _ = eval_env.reset()
            agent_goal = [(agent_state["grid"]["goal"], agent_state["goal"])]
            agent_start = [
                (agent_state["grid"]["observation"], agent_state["observation"])
            ]

            # Add the new observations and goals to the state
            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].extend(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not change ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        search_policy.select_action(state)
        waypoints = search_policy.get_augmented_waypoints()

    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        (
            starts,
            goals,
            _,
            waypoints,
            _,
            _,
        ) = collector_cls.get_trajectories(
            search_policy, eval_env, num_agents, habitat=True, threshold=COLLISION_THRESHOLD
        )

    _, ax = plt.subplots(figsize=(8, 9))
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    normalizing_factor = np.array([height, width])

    agent_waypoints = []
    for agent_id in range(num_agents):

        agent_goal = goals[agent_id]
        agent_goal_grid, agent_goal_visual = agent_goal
        agent_goal_grid = agent_goal_grid / normalizing_factor

        agent_start = starts[agent_id]
        agent_start_grid, _ = agent_start
        agent_start_grid = agent_start_grid / normalizing_factor

        waypoints_grid = (
            np.array([wp[0] for wp in waypoints[agent_id]]) / normalizing_factor
        )
        waypoints_grid = np.array([agent_start_grid, *waypoints_grid, agent_goal_grid])
        agent_waypoints.append(waypoints_grid)

        waypoints_visual = [wp[1] for wp in waypoints[agent_id]]

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start_grid}")
        print(f"Waypoints: {waypoints_grid}")
        print(f"Goal: {agent_goal_grid}")
        print(f"Steps: {waypoints_grid.shape[0] - 1}")
        print("-" * 10)

        ax = plot_agent_paths(
            agent_id,
            agent_start_grid,
            agent_goal_grid,
            waypoints_grid,
            "Search",
            ax,
            obs_marker="s",
        )

        if len(outpath) > 0:
            save_path = outpath[:-4] + f"_agent_{agent_id}" + extension
            visualize_habitat_agent([*waypoints_visual, agent_goal_visual], save_path)

    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=num_agents, fontsize=16
    )
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        save_path = outpath[:-4] + extension
        fig, ax = plt.subplots(figsize=(8, 9))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(agent_waypoints, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_search_path(
    search_policy, eval_env, outpath="", difficulty=0.5, num_agents=None
):
    if num_agents is None:
        visualize_search_path_single_agent(search_policy, eval_env, outpath, difficulty)
    else:
        visualize_search_path_multi_agent(
            search_policy, eval_env, num_agents, outpath, difficulty
        )


def visualize_compare_search_single_agent(
    agent, search_policy, eval_env, seed=0, outpath="", difficulty=0.5
):

    constrained = search_policy.constraints is not None
    set_habitat_env_difficulty(eval_env.env, difficulty)

    plt.figure(figsize=(12, 6))

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        height, width = eval_env.walls.shape
        normalizing_factor = np.array([height, width])

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent

        collector_cls = ConstrainedCollector if constrained else Collector
        if col_index == 0:
            start, goal, observations, waypoints, _, records = (
                collector_cls.get_trajectory(policy, eval_env, habitat=True)
            )
        else:
            start_cost = (
                records["first_step_cost"] if "first_step_cost" in records else None
            )
            _, _, observations, waypoints, _, records = collector_cls.get_trajectory(
                policy,
                eval_env,
                habitat=True,
                input_start=start,
                input_goal=goal,
                start_cost=start_cost,
            )

        goal_grid, goal_visual = goal
        goal_grid = goal_grid / normalizing_factor

        start = observations[0]
        start_grid, _ = start
        start_grid = start_grid / normalizing_factor

        observations_grid = (
            np.array([obs[0] for obs in observations]) / normalizing_factor
        )
        observations_visual = [obs[1] for obs in observations]

        waypoints_grid = np.array([wp[0] for wp in waypoints]) / normalizing_factor
        waypoints_grid = (
            np.array([start_grid, *waypoints_grid, goal_grid]) if use_search else None
        )

        print(f"Policy: {title}")
        print(f"Start: {start_grid}")
        print(f"Goal: {goal_grid}")
        print(f"Reward: {records['rewards']}")
        print(f"Steps: {records['steps']}")
        if "max_step_cost" in records:
            print(f"Max Step Cost: {records['max_step_cost']}")
            print(f"Cumulative Cost: {records['cumulative_costs']}")
        print("-" * 10)

        ax = plot_agent_paths(
            0, start_grid, goal_grid, observations_grid, title, ax, waypoints_grid
        )
        if not use_search:
            ax.legend(
                loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16
            )

        if len(outpath) > 0:
            save_path = (
                outpath[:-4] + "_search" + extension
                if use_search
                else outpath[:-4] + "_no_search" + extension
            )
            visualize_habitat_agent([*observations_visual, goal_visual], save_path)

    plt.suptitle("Single-Agent Search Comparison", fontsize=24)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_compare_search_multi_agent(
    agent, search_policy, eval_env, n_agents, seed=0, outpath="", difficulty=0.5
):

    constrained = search_policy.constraints is not None
    set_habitat_env_difficulty(eval_env.env, difficulty)

    plt.figure(figsize=(12, 6))

    search_waypoints = []
    search_observations = []
    no_search_observations = []

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        height, width = eval_env.walls.shape
        normalizing_factor = np.array([height, width])

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent
        collector_cls = ConstrainedCollector if constrained else Collector

        if col_index == 0:
            starts, goals, observations, waypoints, _, records = (
                collector_cls.get_trajectories(
                    policy, eval_env, n_agents, habitat=True, threshold=COLLISION_THRESHOLD
                )
            )
        else:
            start_costs = []
            for agent_id in range(n_agents):
                if "first_step_cost" in records[agent_id]:
                    start_costs.append(records[agent_id]["first_step_cost"])
                else:
                    start_costs = None
                    break
            _, _, observations, waypoints, _, records = collector_cls.get_trajectories(
                policy,
                eval_env,
                n_agents,
                starts,
                goals,
                habitat=True,
                start_costs=start_costs,
                threshold=COLLISION_THRESHOLD,
            )

        print(f"Policy: {title}")
        for agent_id in range(n_agents):

            agent_goal = goals[agent_id]
            agent_goal_grid, agent_goal_visual = agent_goal
            agent_goal_grid = agent_goal_grid / normalizing_factor

            agent_start = observations[agent_id][0]
            agent_start_grid, _ = agent_start
            agent_start_grid = agent_start_grid / normalizing_factor

            observations_grid = (
                np.array([obs[0] for obs in observations[agent_id]])
                / normalizing_factor
            )
            observations_visual = [obs[1] for obs in observations[agent_id]]

            waypoints_grid = (
                np.array([wp[0] for wp in waypoints[agent_id]]) / normalizing_factor
            )
            waypoints_grid = (
                np.array([agent_start_grid, *waypoints_grid, agent_goal_grid])
                if use_search
                else None
            )

            print(f"Agent: {agent_id}")
            print(f"Start: {agent_start_grid}")
            if use_search:
                search_waypoints.append(waypoints_grid)
                search_observations.append(observations_grid)
                print(f"Waypoints: {waypoints_grid}")
            else:
                no_search_observations.append(observations_grid)
            print(f"Goal: {agent_goal_grid}")
            print(f"Reward: {records[agent_id]['rewards']}")
            print(f"Steps: {records[agent_id]['steps']}")
            if "max_step_cost" in records[agent_id]:
                print(f"Max Step Cost: {records[agent_id]['max_step_cost']}")
                print(f"Cumulative Cost: {records[agent_id]['cumulative_costs']}")
            print("-" * 10)

            ax = plot_agent_paths(
                agent_id,
                agent_start_grid,
                agent_goal_grid,
                observations_grid,
                title,
                ax,
                waypoints_grid,
            )

            if len(outpath) > 0:
                if use_search:
                    save_path = outpath[:-4] + f"_search_agent_{agent_id}" + extension
                else:
                    save_path = (
                        outpath[:-4] + f"_no_search_agent_{agent_id}" + extension
                    )
                visualize_habitat_agent(
                    [*observations_visual, agent_goal_visual], save_path
                )

        if not use_search:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=n_agents,
                fontsize=16,
            )

    plt.suptitle("Multi-Agent Search Comparison", fontsize=24)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)

        save_path = outpath[:-4] + "_search" + extension
        fig, ax = plt.subplots(figsize=(10, 11))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(
            search_observations,
            save_path,
            (fig, ax),
            save_fig=True,
            waypoints=search_waypoints,
        )

        save_path = outpath[:-4] + "_no_search" + extension
        fig, ax = plt.subplots(figsize=(10, 11))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(no_search_observations, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_compare_search(
    agent, search_policy, eval_env, seed=0, outpath="", difficulty=0.5, num_agents=None
):
    if num_agents is None:
        visualize_compare_search_single_agent(
            agent, search_policy, eval_env, seed, outpath, difficulty
        )
    else:
        visualize_compare_search_multi_agent(
            agent, search_policy, eval_env, num_agents, seed, outpath, difficulty
        )
