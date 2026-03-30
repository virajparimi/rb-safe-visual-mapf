"""
Evaluate the accuracy and reliability of reward and cost critics
"""

import time
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from termcolor import cprint
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.data_struct import dict_expand
from pud.algos.policies import GaussianPolicy
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.visualizers.visualize import visualize_eval_records
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import eval_agent_from_Q
from pud.collectors.constrained_collector import ConstrainedCollector as Collector
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
)
from pud.envs.safe_pointenv.pb_sampler import (
    load_pb_set,
    sample_pbs_by_agent,
    sample_cost_pbs_by_agent,
)


def train_eval(
    policy: GaussianPolicy,
    agent: DRLDDPGLag,  # Agent to be trained
    agent_g: DRLDDPGLag,  # Unconstrained reference agent
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
    tensorboard_writer: Optional[SummaryWriter] = None,
    num_eval_episodes: int = 10,
    pbar=True,
    sample_size: int = 100,
    num_train_pbs_per_ref: int = 10,
    cost_min_dist: float = 1.0,
    cost_max_dist: float = 10.0,
    uncertainty_ub: float = 1.0,
    uncertainty_lb: float = 0.0,
    verbose=True,
    illustration_pb_file="",
    ckpt_dir: Path = Path(""),
    vis_dir: Union[Path, None] = Path(""),
):
    """Train constrained RL agent"""
    env.set_verbose(False)  # Too much warn msgs due to empty queue
    env.set_use_q(True)
    agent.set_lag_status(turn_on_lag=True)

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

        # The itr/s shown in tqdm is reduced by a factor of collect_steps
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

            # Use reference agent to generate problem samples
            update_train_pbs_by_metric(
                agent=agent_g,
                env=env,
                num_pbs_per_ref=num_train_pbs_per_ref,
                sample_size=sample_size,
                cost_min_dist=cost_min_dist,
                cost_max_dist=cost_max_dist,
                use_uncertainty=True,
                uncertainty_lb=uncertainty_lb,
                uncertainty_ub=uncertainty_ub,
                illustration_pb_file=illustration_pb_file,
            )

            assert eval_func is not None, "eval_func is not defined"
            assert vis_dir is not None, "vis_dir is not defined"
            eval_info = eval_func(
                agent=agent,
                agent_g=agent_g,
                eval_env=eval_env,
                eval_distances=eval_distances,
                num_evals=num_eval_episodes,
                sample_size=sample_size,
                illustration_pb_file=illustration_pb_file,
                vis_dir=vis_dir.joinpath("itr_{:0>6d}".format(i)),
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
                # For dists
                field_header = "Eval Dist ~ "
                for ii in eval_info["dists"]:
                    tensorboard_writer.add_scalars(
                        field_header
                        + "{:0>2d}/mean".format(eval_info["dists"][ii]["agent"]["ref"]),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info["dists"][ii]["agent"]["pred"]),
                            "val": -1.0
                            * np.mean(eval_info["dists"][ii]["agent"]["vals"]),
                            "pred_greedy": np.mean(
                                eval_info["dists"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": -1.0
                            * np.mean(eval_info["dists"][ii]["agent_g"]["vals"]),
                        },
                        global_step=i,
                    )

                    tensorboard_writer.add_scalars(
                        field_header
                        + "{:0>2d}/std".format(eval_info["dists"][ii]["agent"]["ref"]),
                        tag_scalar_dict={
                            "pred": np.std(eval_info["dists"][ii]["agent"]["pred"]),
                            "val": np.std(eval_info["dists"][ii]["agent"]["vals"]),
                            "pred_greedy": np.std(
                                eval_info["dists"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": np.std(
                                eval_info["dists"][ii]["agent_g"]["vals"]
                            ),
                        },
                        global_step=i,
                    )

                    N_success = np.array(
                        eval_info["dists"][ii]["agent"]["success"], dtype=float
                    )
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        tensorboard_writer.add_scalar(
                            field_header
                            + "{:0>2d}/success_rate".format(
                                eval_info["dists"][ii]["agent"]["ref"]
                            ),
                            success_rate,
                            global_step=i,
                        )

                field_header = "Eval Cost ~ "
                for ii in eval_info["costs"]:
                    tensorboard_writer.add_scalars(
                        field_header + "{}/mean".format(ii),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info["costs"][ii]["agent"]["pred"]),
                            "val": np.mean(eval_info["costs"][ii]["agent"]["vals"]),
                            "pred_greedy": np.mean(
                                eval_info["costs"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": np.mean(
                                eval_info["costs"][ii]["agent_g"]["vals"]
                            ),
                        },
                        global_step=i,
                    )
                    tensorboard_writer.add_scalars(
                        field_header + "{}/std".format(ii),
                        tag_scalar_dict={
                            "pred": np.std(eval_info["costs"][ii]["agent"]["pred"]),
                            "val": np.std(eval_info["costs"][ii]["agent"]["vals"]),
                            "pred_greedy": np.std(
                                eval_info["costs"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": np.std(
                                eval_info["costs"][ii]["agent_g"]["vals"]
                            ),
                        },
                        global_step=i,
                    )

                    N_success = np.array(
                        eval_info["costs"][ii]["agent"]["success"], dtype=float
                    )
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        tensorboard_writer.add_scalar(
                            field_header + "{}/success_rate".format(ii),
                            success_rate,
                            global_step=i,
                        )
                    tensorboard_writer.add_scalar(
                        field_header + "{}/N".format(ii), len(N_success), global_step=i
                    )

                # Reset timer
                t_mark = time.time()


def update_train_pbs_by_metric(
    agent: DRLDDPGLag,
    env: SafeGoalConditionedPointQueueWrapper,
    num_pbs_per_ref: int,
    sample_size: int,
    cost_min_dist: float = 1.0,
    cost_max_dist: float = 10.0,
    use_uncertainty: bool = True,  # Boost samples of high uncertainty
    uncertainty_lb: float = 0.0,
    uncertainty_ub: float = 1.0,
):
    # Update pbs in the train eval
    new_train_pbs = []
    target_cost = np.random.uniform(low=agent.lagrange.cost_limit, high=4.0)
    cost_eval_pbs = sample_cost_pbs_by_agent(
        env=env,
        agent=agent,
        num_states=sample_size,
        K=num_pbs_per_ref,
        target_val=target_cost,
        min_dist=cost_min_dist,
        max_dist=cost_max_dist,
        ensemble_agg="mean",
        use_uncertainty=use_uncertainty,
        uncertainty_lb=uncertainty_lb,
        uncertainty_ub=uncertainty_ub,
    )
    new_train_pbs.extend(cost_eval_pbs)
    env.set_pbs(pb_list=new_train_pbs)


def eval_agent_by_metric(
    agent: DRLDDPGLag,
    eval_env: SafeGoalConditionedPointQueueWrapper,
    num_evals: int,
    sample_size: int,
    target_val: float,
    ensemble_agg: str,
):
    assert num_evals <= sample_size
    pbs = sample_pbs_by_agent(
        env=eval_env,
        agent=agent,
        num_states=sample_size,
        target_val=target_val,
        K=num_evals,
        min_dist=target_val,
        max_dist=target_val,
        use_uncertainty=False,
        ensemble_agg=ensemble_agg,
    )
    eval_env.append_pbs(pb_list=pbs)  # type: ignore
    eval_stats = eval_agent_from_Q(policy=agent, eval_env=eval_env)
    return eval_stats


def gather_log(eval_stats: dict, names_n_keys: Dict[str, list]):
    """
    Eval_stats has the form of eval_stats[order_id][rest of keys]
    names_n_keys offers the list of keys to read data from eval_stats[id], and
        defines a convenient name, e.g.,
        "name", ["init_info","prediction"]
    """
    logs = {}
    for n in names_n_keys.keys():
        logs[n] = []

    for id in eval_stats.keys():
        for n in names_n_keys.keys():
            logs[n].append(dict_expand(D=eval_stats[id], keys=names_n_keys[n]))
    return logs


def eval_pointenv_cost_constrained_dists(
    agent: DRLDDPGLag,
    agent_g: DRLDDPGLag,
    eval_env: SafeGoalConditionedPointWrapper,
    illustration_pb_file: str,
    num_evals: int = 10,
    sample_size: int = 100,
    eval_distances=[2, 5, 10],
    vis_dir: Optional[Path] = None,
):
    collect_trajs = False
    if vis_dir is not None:
        collect_trajs = True

    dist_eval_stats = dict()
    for ii_d in range(len(eval_distances)):
        pbs = sample_pbs_by_agent(
            env=eval_env,
            agent=agent_g,
            num_states=sample_size,
            target_val=eval_distances[ii_d],
            K=num_evals,
            min_dist=0,
            max_dist=20,
            use_uncertainty=False,
            ensemble_agg="mean",
        )
        if len(pbs) > 0:
            eval_env.append_pbs(pb_list=deepcopy(pbs))
            dist_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=collect_trajs,
            )
            eval_env.append_pbs(pb_list=deepcopy(pbs))
            dist_eval_i_g = eval_agent_from_Q(
                policy=agent_g,
                eval_env=eval_env,
                collect_trajs=collect_trajs,
            )
            if collect_trajs:
                assert vis_dir is not None, "vis_dir is not defined"
                vis_dir.mkdir(parents=True, exist_ok=True)
                start_list = [p["start"].tolist() for p in pbs]
                goal_list = [p["goal"].tolist() for p in pbs]
                fig, ax = plt.subplots()
                ax = visualize_eval_records(
                    eval_records=dist_eval_i,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    color="g",
                )
                ax = visualize_eval_records(
                    eval_records=dist_eval_i_g,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    color="r",
                )
                ax.legend()
                ax.set_title("target dist ~ {}".format(eval_distances[ii_d]))
                figname = "dist={:0>2d}.jpg".format(eval_distances[ii_d])
                fig.savefig(vis_dir.joinpath(figname), dpi=300)
                plt.close(fig=fig)
            dist_eval_stats[ii_d] = {}
            dist_logs = gather_log(
                eval_stats=dist_eval_i,
                names_n_keys={
                    "attr_vals": ["rewards"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            dist_eval_stats[ii_d]["agent"] = {
                "vals": dist_logs["attr_vals"],
                "pred": dist_logs["attr_pred"],
                "ref": eval_distances[ii_d],
                "success": dist_logs["success_hist"],
            }
            dist_logs_g = gather_log(
                eval_stats=dist_eval_i_g,
                names_n_keys={
                    "attr_vals": ["rewards"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            dist_eval_stats[ii_d]["agent_g"] = {
                "vals": dist_logs_g["attr_vals"],
                "pred": dist_logs_g["attr_pred"],
                "ref": eval_distances[ii_d],
                "success": dist_logs_g["success_hist"],
            }
        else:
            print("[WARN] empty set for dist eval problem")

    cost_eval_stats = dict()
    cost_eval_pbs = load_pb_set(
        file_path=illustration_pb_file,
        env=eval_env,
        agent=agent,
    )
    if len(cost_eval_pbs) > 0:
        eval_env.append_pbs(pb_list=deepcopy(cost_eval_pbs))
        cost_eval_i = eval_agent_from_Q(
            policy=agent,
            eval_env=eval_env,
            collect_trajs=collect_trajs,
        )
        eval_env.append_pbs(pb_list=deepcopy(cost_eval_pbs))
        cost_eval_i_g = eval_agent_from_Q(
            policy=agent_g,
            eval_env=eval_env,
            collect_trajs=collect_trajs,
        )
        if collect_trajs:
            assert vis_dir is not None, "vis_dir is not defined"
            vis_dir.mkdir(parents=True, exist_ok=True)
            start_list = [
                p["start"].tolist() if p is not None else None for p in cost_eval_pbs
            ]
            goal_list = [
                p["goal"].tolist() if p is not None else None for p in cost_eval_pbs
            ]
            fig, ax = plt.subplots()
            ax = visualize_eval_records(
                eval_records=cost_eval_i,
                eval_env=eval_env,
                ax=ax,
                starts=start_list,
                goals=goal_list,
                color="g",
            )
            ax = visualize_eval_records(
                eval_records=cost_eval_i_g,
                eval_env=eval_env,
                ax=ax,
                starts=start_list,
                goals=goal_list,
                color="r",
            )
            for jj in range(len(start_list)):
                xy_n = eval_env.normalize_obs(start_list[jj])
                ax.text(
                    x=xy_n[0] + 0.1,
                    y=xy_n[1],
                    s="cost={:.2f}, cost greedy={:.2f}".format(
                        cost_eval_i[jj]["cum_costs"], cost_eval_i_g[jj]["cum_costs"]
                    ),
                )

            ax.set_title("Illustration problem comparison")
            ax.legend()
            figname = "ref.jpg"
            fig.savefig(vis_dir.joinpath(figname), dpi=300)
            plt.close(fig=fig)

        cost_eval_stats["ref"] = {}
        cost_logs = gather_log(
            eval_stats=cost_eval_i,
            names_n_keys={
                "attr_vals": ["cum_costs"],
                "attr_pred": ["init_info", "prediction"],
                "success_hist": ["success"],
            },
        )
        cost_eval_stats["ref"]["agent"] = {
            "vals": cost_logs["attr_vals"],
            "pred": cost_logs["attr_pred"],
            "success": cost_logs["success_hist"],
        }
        cost_logs_g = gather_log(
            eval_stats=cost_eval_i_g,
            names_n_keys={
                "attr_vals": ["cum_costs"],
                "attr_pred": ["init_info", "prediction"],
                "success_hist": ["success"],
            },
        )
        cost_eval_stats["ref"]["agent_g"] = {
            "vals": cost_logs_g["attr_vals"],
            "pred": cost_logs_g["attr_pred"],
            "success": cost_logs_g["success_hist"],
        }

    eval_stats = {}
    eval_stats["dists"] = dist_eval_stats
    eval_stats["costs"] = cost_eval_stats
    return eval_stats
