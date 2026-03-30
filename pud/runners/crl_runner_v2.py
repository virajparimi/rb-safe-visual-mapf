"""
Evaluate the accuracy and reliability of reward and cost critics
"""

import time
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional
from termcolor import cprint
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import GaussianPolicy
from pud.algos.data_struct import init_embedded_dict
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import ConstrainedCollector as Collector
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper


def train_eval(
    policy: GaussianPolicy,
    agent: DRLDDPGLag,
    replay_buffer: ConstrainedReplayBuffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=2,
    opt_steps=1,
    batch_size_opt=64,
    eval_func=None,  # Make this a partial func
    opt_log_interval=100,
    eval_interval=10000,
    eval_distances=[2, 5, 10],  # Reference grouping based on estimated distances
    eval_cost_intervals=[0.0, 0.2, 0.5, 1.0],  # Grouping cost eval results
    tensorboard_writer: Optional[SummaryWriter] = None,
    warmup_epochs: int = 100,
    num_eval_episodes: int = 10,
    pbar=True,
    verbose=True,
    ckpt_dir: Path = Path(""),
):
    """Train constrained RL agent"""
    collector = Collector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )

    num_eps = collector.num_eps
    ep_cost = 0.0
    collector.step(collector.initial_collect_steps)

    pbar = tqdm(total=num_iterations, disable=not pbar)
    t_mark = time.time()
    for i in range(1, num_iterations + 1):
        pbar.update()

        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(
            replay_buffer, iterations=opt_steps, batch_size=batch_size_opt
        )

        if collector.num_eps > num_eps:
            ep_cost = collector.past_eps[-1]["ep_cost"]
            ep_len = collector.past_eps[-1]["ep_len"]
            if verbose:
                cprint(
                    "[INFO] eps Jc={:.2f}, eps length={}".format(ep_cost, ep_len),
                    "green",
                )
            num_eps = collector.num_eps

            if i > warmup_epochs:
                agent.optimize_lagrange(ep_cost=ep_cost)

        if i % opt_log_interval == 0:
            if verbose:
                print(f"iteration = {i}, opt_info = {opt_info}")

        if i % eval_interval == 0:
            if isinstance(ckpt_dir, Path):
                torch.save(
                    agent.state_dict(), ckpt_dir.joinpath("ckpt_{:0>7d}".format(i))
                )

            agent.eval()
            if verbose:
                print(f"evaluating iteration = {i}")

            assert eval_func is not None, "eval_func must be provided!"
            eval_info = eval_func(
                agent=agent,
                eval_env=eval_env,
                eval_distances=eval_distances,
                num_evals=num_eval_episodes,
                cost_intervals=eval_cost_intervals,
            )
            if verbose:
                print("-" * 10)

        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i
            )
            tensorboard_writer.add_scalar(
                "Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i
            )
            tensorboard_writer.add_scalar(
                "Opt/cost_critic_loss",
                np.mean(opt_info["cost_critic_loss"]),
                global_step=i,
            )
            tensorboard_writer.add_scalar(
                "Opt/Lagrange_Multiplier",
                agent.lagrange.lagrangian_multiplier.item(),
                global_step=i,
            )

            if i % eval_interval == 0:
                rate = float(eval_interval) / (time.time() - t_mark)
                tensorboard_writer.add_scalar(
                    "Opt/Rate(Iter per sec)", rate, global_step=i
                )

                for d_ref in eval_info:
                    for c_ref in eval_info[d_ref]:
                        field_header = "Eval_D={:0>2d} C={:.2f}".format(d_ref, c_ref)
                        # logging for distance prediction
                        tensorboard_writer.add_scalar(
                            field_header + "/d_pred_mean",
                            np.mean(eval_info[d_ref][c_ref]["r"]["pred"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/d_pred_std",
                            np.std(eval_info[d_ref][c_ref]["r"]["pred"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/d_true_mean",
                            np.mean(eval_info[d_ref][c_ref]["r"]["true"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/d_true_std",
                            np.std(eval_info[d_ref][c_ref]["r"]["true"]),
                            global_step=i,
                        )

                        # Logging for cost prediction
                        tensorboard_writer.add_scalar(
                            field_header + "/c_pred_mean",
                            np.mean(eval_info[d_ref][c_ref]["c"]["pred"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/c_pred_std",
                            np.std(eval_info[d_ref][c_ref]["c"]["pred"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/c_true_mean",
                            np.mean(eval_info[d_ref][c_ref]["c"]["true"]),
                            global_step=i,
                        )
                        tensorboard_writer.add_scalar(
                            field_header + "/c_true_std",
                            np.std(eval_info[d_ref][c_ref]["c"]["true"]),
                            global_step=i,
                        )

                # Reset timer
                t_mark = time.time()


def eval_pointenv_cost_constrained_dists(
    agent,
    eval_env: SafeGoalConditionedPointWrapper,
    num_evals=10,
    eval_distances=[2, 5, 10, 20],
    cost_intervals=[0.0, 0.2, 0.5, 1.0],
):
    """Sample starts and goals that are lower than a preset maximum cost limit (linear interpolation).
    The reference apsp in eval_env are all integers because they are computed from grid distances (no diagonal paths)
    """
    # eval_stats: ref_dict -> ref_cost -> {pred, true}
    eval_stats = dict()

    for idx_d in range(len(eval_distances)):
        min_dist, max_dist = eval_distances[idx_d], eval_distances[idx_d]
        for idx_c in range(len(cost_intervals)):
            min_cost, max_cost = cost_intervals[idx_c], cost_intervals[idx_c]
            eval_env.set_sample_goal_args(
                prob_constraint=1,
                min_dist=min_dist,
                max_dist=max_dist,
                min_cost=min_cost,
                max_cost=max_cost,
            )

            eval_outputs = Collector.eval_agent_n_record_init_states(
                agent, eval_env, num_evals
            )

            # Estimate distance-to-goal from initial states
            states = dict(observation=[], goal=[])
            dist_from_rewards = (
                []
            )  # Not ground truth distance, but should be accurate when policy is trained
            ep_costs = []
            for key in eval_outputs.keys():
                states["observation"].append(
                    eval_outputs[key]["init_states"]["observation"]
                )
                states["goal"].append(eval_outputs[key]["init_states"]["goal"])
                dist_from_rewards.append(-eval_outputs[key]["rewards"])
                ep_costs.append(eval_outputs[key]["costs"])

            pred_dist = list(agent.get_dist_to_goal(states))
            pred_costs = list(agent.get_cost_to_goal(states))

            init_embedded_dict(
                eval_stats,
                embeds=[
                    (min_dist, dict),  # ref dict
                    (min_cost, dict),  # ref cost
                    ("r", dict),
                ],
            )
            init_embedded_dict(
                eval_stats,
                embeds=[
                    (min_dist, dict),  # ref dict
                    (min_cost, dict),  # ref cost
                    ("c", dict),
                ],
            )

            eval_stats[min_dist][min_cost]["r"]["pred"] = pred_dist
            eval_stats[min_dist][min_cost]["r"]["true"] = dist_from_rewards
            eval_stats[min_dist][min_cost]["c"]["pred"] = pred_costs
            eval_stats[min_dist][min_cost]["c"]["true"] = ep_costs

    return eval_stats
