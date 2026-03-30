import time
import numpy as np
from dotmap import DotMap
from tqdm.auto import tqdm
from typing import Optional

from pud.collectors.collector import Collector
from torch.utils.tensorboard.writer import SummaryWriter
from pud.envs.simple_navigation_env import set_env_difficulty
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper,
    SafeTimeLimit,
    set_safe_env_difficulty,
)


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
    tensorboard_writer: Optional[SummaryWriter] = None,
    verbose=True,
):
    collector = Collector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )
    collector.step(collector.initial_collect_steps)
    for i in tqdm(range(1, num_iterations + 1), total=num_iterations):
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(
            replay_buffer, iterations=opt_steps, batch_size=batch_size_opt
        )

        if i % opt_log_interval == 0:
            if verbose:
                print(f"iteration = {i}, opt_info = {opt_info}")

        if i % eval_interval == 0:
            agent.eval()
            if verbose:
                print(f"evaluating iteration = {i}")
            eval_info = eval_func(agent, eval_env)
            if verbose:
                print("-" * 10)

        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i
            )
            tensorboard_writer.add_scalar(
                "Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i
            )

            if i % eval_interval == 0:
                field_header = "Eval Dist ~ "
                for d_ref in eval_info:
                    tensorboard_writer.add_scalars(
                        field_header + "{:0>2d}/mean".format(d_ref),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info[d_ref]["pred_dist"]),
                            "val": -np.mean(eval_info[d_ref]["returns"]),
                        },
                        global_step=i,
                    )

                    tensorboard_writer.add_scalars(
                        field_header + "{:0>2d}/std".format(d_ref),
                        tag_scalar_dict={
                            "pred": np.std(eval_info[d_ref]["pred_dist"]),
                            "val": -np.std(eval_info[d_ref]["returns"]),
                        },
                        global_step=i,
                    )


def eval_pointenv_dists(
    agent, eval_env, num_evals=10, eval_distances=[2, 5, 10], verbose=True
):
    eval_info = DotMap()
    for dist in eval_distances:
        eval_env.set_sample_goal_args(
            prob_constraint=1, min_dist=dist, max_dist=dist
        )  # NOTE: Samples goal distances in [min_dist, max_dist] closed interval
        returns = Collector.eval_agent(agent, eval_env, num_evals)
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
            print(f"\t\treturns = {returns}")
            print(f"\t\tpredicted_dists = {pred_dist}")
            print(f"\t\taverage return = {np.mean(returns)}")
            print(
                f"\t\taverage predicted_dist = {np.mean(pred_dist):.1f} ({np.std(pred_dist):.2f})"
            )

        eval_info[dist]["pred_dist"] = pred_dist
        eval_info[dist]["returns"] = returns
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
