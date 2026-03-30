import time
from typing import Union
import torch
import numpy as np
from pathlib import Path
from dotmap import DotMap
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from pud.collectors.collector import Collector
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.collectors.vec_collector import VectorCollector
from pud.collectors.visual_collector import VisualCollector
from pud.envs.simple_navigation_env import set_env_difficulty
from pud.envs.habitat_navigation_env import plot_wall, plot_traj
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper,
    SafeTimeLimit,
    set_safe_env_difficulty,
)

# Generated according to https://medialab.github.io/iwanthue/
distinct_colors = (
    np.array(
        [
            [190, 121, 68],
            [176, 86, 193],
            [104, 184, 84],
            [105, 125, 204],
            [190, 168, 66],
            [191, 113, 172],
            [98, 128, 63],
            [201, 80, 113],
            [77, 184, 175],
            [211, 82, 56],
        ],
        dtype=float,
    )
    / 255
)


def log_time(step: int = 0, log: Union[dict, None] = None):
    if log is None:
        log = {
            "time": [time.time()],
            "step": [step],
            "speed": [],
        }
        return log

    log["time"].append(time.time())
    log["step"].append(step)
    log["speed"].append(
        float(log["step"][-1] - log["step"][-2])
        / (float(log["time"][-1]) - float(log["time"][-2]))
    )
    return log


def train_eval(
    policy,
    agent,
    replay_buffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=1,
    opt_steps=1,
    batch_size_opt=64,
    eval_func=lambda agent, eval_env: None,
    opt_log_interval=100,
    eval_interval=10000,
    verbose=True,
    logger: dict = {},
):
    time_logs = log_time(step=0)
    collector = VisualCollector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )
    collector.step(collector.initial_collect_steps)
    for i in tqdm(range(1, num_iterations + 1), total=num_iterations):
        logger["i"] = i
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(
            replay_buffer, iterations=opt_steps, batch_size=batch_size_opt
        )

        if i % opt_log_interval == 0:
            if verbose:
                print(f"iteration = {i}, opt_info = {opt_info}")

        if i % eval_interval == 0:
            if "ckpt" in logger and isinstance(logger["ckpt"], Path):
                torch.save(
                    agent.state_dict(),
                    logger["ckpt"].joinpath("ckpt_{:0>7d}".format(i)),
                )

            agent.eval()
            if verbose:
                print(f"evaluating iteration = {i}")
            eval_info = eval_func(
                agent,
                eval_env,
                logger=logger,
            )
            if verbose:
                print("-" * 10)

        if "tb" in logger:
            if i > 1 and i % opt_log_interval == 0:
                logger["tb"].add_scalar(
                    "Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i
                )
                logger["tb"].add_scalar(
                    "Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i
                )

            if i > 1 and i % eval_interval == 0:
                field_header = "Eval Dist ~ "
                for d_ref in eval_info:
                    logger["tb"].add_scalars(
                        field_header + "{:0>2d}/mean".format(d_ref),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info[d_ref]["pred_dist"]),
                            "val": -np.mean(eval_info[d_ref]["returns"]),
                        },
                        global_step=i,
                    )

                    logger["tb"].add_scalars(
                        field_header + "{:0>2d}/std".format(d_ref),
                        tag_scalar_dict={
                            "pred": np.std(eval_info[d_ref]["pred_dist"]),
                            "val": -np.std(eval_info[d_ref]["returns"]),
                        },
                        global_step=i,
                    )

                    N_success = np.array(eval_info[d_ref]["success"], dtype=float)
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        logger["tb"].add_scalar(
                            field_header + "{:0>2d}/success_rate".format(d_ref),
                            success_rate,
                            global_step=i,
                        )

                time_logs = log_time(step=i, log=time_logs)
                logger["tb"].add_scalar(
                    "Time/Iters per Seconds", time_logs["speed"][-1], global_step=i
                )
                logger["tb"].add_scalar(
                    "Time/Total Time", time_logs["time"][-1], global_step=i
                )


def eval_pointenv_dists(
    agent,
    eval_env,
    num_evals=10,
    eval_distances=[1, 2, 3, 4],
    verbose=True,
    logger: dict = {},
):
    eval_info = DotMap()
    if "eval_distances" in logger:
        eval_distances = logger["eval_distances"]

    eval_img_dir: Union[Path, None] = None
    if "imgs" in logger:
        eval_img_dir = logger["imgs"].joinpath("eval_{:0>5d}".format(logger["i"]))
        assert eval_img_dir is not None
        eval_img_dir.mkdir(exist_ok=True, parents=True)

    for dist in eval_distances:
        eval_env.set_sample_goal_args(
            prob_constraint=1, min_dist=dist, max_dist=dist
        )  # NOTE: Samples goal distances in [min_dist, max_dist] closed interval
        outs = VectorCollector.eval_agent_n_trajs(agent, eval_env, num_evals)

        if eval_img_dir:
            fig, ax = plt.subplots()
            normalize_map = False
            ax = plot_wall(eval_env.walls.copy(), ax, normalize=normalize_map)
            goals = []
            for ii in range(len(outs["trajs"])):
                goals.append(outs["trajs"][ii][0]["grid"]["goal"])
            goals = np.stack(goals, axis=0)

            def get_traj(inp_traj):
                return [
                    inp_traj[ii]["grid"]["observation"] for ii in range(len(inp_traj))
                ]

            for ii, tt in enumerate(outs["trajs"]):
                cur_traj = np.stack(get_traj(tt), axis=0)
                ax = plot_traj(
                    traj=cur_traj,
                    walls=eval_env.walls.copy(),
                    normalize=normalize_map,
                    ax=ax,
                    color=distinct_colors[ii],
                    label="traj{:0>2d}".format(ii),
                    marker="o",
                    markersize=4,
                )
                ax.scatter(
                    goals[ii : ii + 1, 0],  # noqa
                    goals[ii : ii + 1, 1],  # noqa
                    marker="*",
                    s=12,
                    color=distinct_colors[ii],
                )
            fig.savefig(eval_img_dir.joinpath("dist={}".format(dist)), dpi=300)
            plt.close(fig=fig)

        # For debugging, it's helpful to check the predicted distances for
        # goals of known distance.
        states = dict(observation=[], goal=[])
        for _ in range(num_evals):
            state = eval_env.reset()
            states["observation"].append(state["observation"])
            states["goal"].append(state["goal"])
        pred_dist = list(agent.get_dist_to_goal(states))

        if verbose:
            print(f"\tset goal dist = {dist}")
            print("\t\treturns = {}".format(outs["rewards"]))
            print(f"\t\tpredicted_dists = {pred_dist}")
            print("\t\taverage return = {}".format(np.mean(outs["rewards"])))
            print(
                f"\t\taverage predicted_dist = {np.mean(pred_dist):.1f} ({np.std(pred_dist):.2f})"
            )

        eval_info[dist]["pred_dist"] = pred_dist
        eval_info[dist]["returns"] = outs["rewards"]
        eval_info[dist]["success"] = outs["success"]
    return eval_info


def eval_search_policy(search_policy, eval_env, num_evals=10, constrained=False):
    eval_start = time.perf_counter()

    successes = 0.0
    for _ in range(num_evals):
        try:
            if constrained:
                _, _, _, _, ep_reward_list, _ = ConstrainedCollector.get_trajectory(
                    search_policy, eval_env
                )
            else:
                _, _, _, _, ep_reward_list, _ = Collector.get_trajectory(
                    search_policy, eval_env
                )
            successes += int(len(ep_reward_list) < eval_env.duration)
        except Exception:
            pass

    eval_end = time.perf_counter()
    eval_time = eval_end - eval_start
    success_rate = successes / num_evals
    return success_rate, eval_time


def take_cleanup_steps(
    search_policy,
    eval_env,
    num_cleanup_steps,
    cost_constraints: dict = {},
    constrained=False,
):
    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, 0.95, **cost_constraints)
    else:
        set_env_difficulty(eval_env, 0.95)

    search_policy.set_cleanup(True)
    cleanup_start = time.perf_counter()
    if constrained:
        ConstrainedCollector.step_cleanup(search_policy, eval_env, num_cleanup_steps)
    else:
        Collector.step_cleanup(
            search_policy, eval_env, num_cleanup_steps
        )  # Samples goals from nodes in state graph
    cleanup_end = time.perf_counter()
    search_policy.set_cleanup(False)
    cleanup_time = cleanup_end - cleanup_start
    return cleanup_time


def cleanup_and_eval_search_policy(
    search_policy,
    eval_env,
    num_evals=10,
    difficulty=0.5,
    cost_constraints: dict = {},
    constrained=False,
):

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )

    # Initial sparse graph
    print(
        f"Initial {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    initial_g, initial_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Filter search policy
    search_policy.filter_keep_k_nearest()

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )
    print(
        f"Filtered {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    filtered_g, filtered_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Cleanup steps
    num_cleanup_steps = int(1e4)
    cleanup_time = take_cleanup_steps(
        search_policy, eval_env, num_cleanup_steps, constrained=constrained
    )
    print(f"Took {num_cleanup_steps} cleanup steps in {cleanup_time:.2f} seconds")

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )
    print(
        f"Cleaned {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    cleaned_g, cleaned_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    return (initial_g, initial_rb), (filtered_g, filtered_rb), (cleaned_g, cleaned_rb)
