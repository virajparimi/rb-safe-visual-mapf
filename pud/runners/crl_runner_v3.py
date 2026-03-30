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
from typing import Optional, Union
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.data_struct import gather_log
from pud.algos.policies import GaussianPolicy
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.visualizers.visualize import visualize_eval_records
from pud.collectors.constrained_collector import eval_agent_from_Q
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
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
    eval_cost_intervals=[0.0, 0.5, 1.0],  # Grouping cost eval results
    tensorboard_writer: Optional[SummaryWriter] = None,
    num_eval_episodes: int = 10,
    pbar=True,
    sample_size: int = 100,
    num_train_pbs_per_ref: int = 10,
    cost_min_dist: float = 1.0,
    cost_max_dist: float = 10.0,
    uncertainty_ub: float = 1.0,
    uncertainty_lb: float = 0.0,
    turn_on_lag: bool = False,
    verbose=True,
    ckpt_dir: Path = Path(""),
    vis_dir: Union[Path, None] = Path(""),
    illustration_pb_file="",
    i_start: int = 1,
    warmup_epochs: int = 200,
    use_uncertainty: bool = True,
):
    """Train constrained RL agent"""
    env.set_verbose(False)  # Too much warn msgs due to empty queue
    env.set_use_q(True)
    agent.set_lag_status(turn_on_lag=turn_on_lag)

    collector = Collector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )

    num_eps = collector.num_eps
    ep_cost = 0.0
    collector.step(collector.initial_collect_steps)

    pbar = tqdm(total=num_iterations, disable=not pbar)
    t_mark = time.time()
    for i in range(i_start, i_start + num_iterations):
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

            # Inject diverse training pbs according to self-evaluation
            update_train_pbs_by_metric(
                agent=agent,
                env=env,
                num_pbs_per_ref=num_train_pbs_per_ref,
                sample_size=sample_size,
                cost_min_dist=cost_min_dist,
                cost_max_dist=cost_max_dist,
                use_uncertainty=True,
                uncertainty_lb=uncertainty_lb,
                uncertainty_ub=uncertainty_ub,
            )

            assert eval_func is not None, "eval_func is not defined!"
            assert vis_dir is not None, "vis_dir is not defined!"
            eval_info = eval_func(
                agent=agent,
                eval_env=eval_env,
                eval_distances=eval_distances,
                num_evals=num_eval_episodes,
                cost_intervals=eval_cost_intervals,
                sample_size=sample_size,
                cost_min_dist=cost_min_dist,
                cost_max_dist=cost_max_dist,
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
                        + "{:0>2d}/mean".format(eval_info["dists"][ii]["ref"]),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info["dists"][ii]["pred"]),
                            "val": -1.0 * np.mean(eval_info["dists"][ii]["vals"]),
                        },
                        global_step=i,
                    )

                    tensorboard_writer.add_scalars(
                        field_header
                        + "{:0>2d}/std".format(eval_info["dists"][ii]["ref"]),
                        tag_scalar_dict={
                            "pred": np.std(eval_info["dists"][ii]["pred"]),
                            "val": np.std(eval_info["dists"][ii]["vals"]),
                        },
                        global_step=i,
                    )

                    N_success = np.array(eval_info["dists"][ii]["success"], dtype=float)
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        tensorboard_writer.add_scalar(
                            field_header
                            + "{:0>2d}/success_rate".format(
                                eval_info["dists"][ii]["ref"]
                            ),
                            success_rate,
                            global_step=i,
                        )

                field_header = "Eval Cost ~ "
                for ii in eval_info["costs"]:
                    tensorboard_writer.add_scalars(
                        field_header
                        + "{:.2f}/mean".format(eval_info["costs"][ii]["ref"]),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info["costs"][ii]["pred"]),
                            "val": np.mean(eval_info["costs"][ii]["vals"]),
                        },
                        global_step=i,
                    )
                    tensorboard_writer.add_scalars(
                        field_header
                        + "{:.2f}/std".format(eval_info["costs"][ii]["ref"]),
                        tag_scalar_dict={
                            "pred": np.std(eval_info["costs"][ii]["pred"]),
                            "val": np.std(eval_info["costs"][ii]["vals"]),
                        },
                        global_step=i,
                    )

                    N_success = np.array(eval_info["costs"][ii]["success"], dtype=float)
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        tensorboard_writer.add_scalar(
                            field_header
                            + "{:.2f}/success_rate".format(
                                eval_info["costs"][ii]["ref"]
                            ),
                            success_rate,
                            global_step=i,
                        )
                    tensorboard_writer.add_scalar(
                        field_header + "{:.2f}/N".format(eval_info["costs"][ii]["ref"]),
                        len(N_success),
                        global_step=i,
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
    # update problems in the train eval
    new_train_pbs = []
    pbs_i = sample_pbs_by_agent(
        env=env,
        agent=agent,
        num_states=sample_size,
        target_val=None,
        K=num_pbs_per_ref,
        ensemble_agg="mean",
        min_dist=0.0,
        max_dist=20.0,
        use_uncertainty=use_uncertainty,  # Boost samples of high uncertainty
        uncertainty_lb=uncertainty_lb,
        uncertainty_ub=uncertainty_ub,
    )
    new_train_pbs.extend(pbs_i)

    cost_eval_pbs = sample_cost_pbs_by_agent(
        env=env,
        agent=agent,
        num_states=sample_size,
        K=num_pbs_per_ref,
        target_val=None,
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


def eval_pointenv_cost_constrained_dists(
    agent,
    eval_env: SafeGoalConditionedPointWrapper,
    num_evals: int = 10,
    sample_size: int = 100,
    eval_distances=[2, 5, 10, 20],
    cost_intervals=[0.0, 0.2, 0.5, 1.0],
    cost_min_dist: float = 0.0,
    cost_max_dist: float = 10.0,
    illustration_pb_file: str = "",
    vis_dir: Optional[Path] = None,
):
    collect_trajs = False
    if vis_dir is not None:
        collect_trajs = True

    dist_eval_stats = dict()

    for ii_d in range(len(eval_distances)):
        pbs = sample_pbs_by_agent(
            env=eval_env,
            agent=agent,
            num_states=sample_size,
            target_val=eval_distances[ii_d],
            K=num_evals,
            min_dist=0,
            max_dist=20,
            use_uncertainty=False,
            ensemble_agg="mean",
        )
        if len(pbs) > 0:
            eval_env.append_pbs(pb_list=pbs)
            dist_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=collect_trajs,
            )
            if collect_trajs:
                assert vis_dir is not None, "vis_dir is not defined!"
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
                )
                ax.legend()
                figname = "dist={:0>2d}.jpg".format(eval_distances[ii_d])
                fig.savefig(vis_dir.joinpath(figname), dpi=300)
                plt.close(fig=fig)

            dist_logs = gather_log(
                eval_stats=dist_eval_i,
                names_n_keys={
                    "attr_vals": ["rewards"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            dist_eval_stats[ii_d] = {
                "vals": dist_logs["attr_vals"],
                "pred": dist_logs["attr_pred"],
                "ref": eval_distances[ii_d],
                "success": dist_logs["success_hist"],
            }
        else:
            print("[WARN] empty set for dist eval problem")

    cost_eval_stats = dict()
    for ii in range(len(cost_intervals)):
        cost_eval_pbs = sample_cost_pbs_by_agent(
            env=eval_env,
            agent=agent,
            num_states=sample_size,
            K=num_evals,
            target_val=cost_intervals[ii],
            min_dist=cost_min_dist,
            max_dist=cost_max_dist,
            use_uncertainty=False,
            ensemble_agg="mean",
        )
        if len(cost_eval_pbs) > 0:
            eval_env.append_pbs(pb_list=cost_eval_pbs)
            cost_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=collect_trajs,
            )
            if collect_trajs:
                assert vis_dir is not None, "vis_dir is not defined!"
                vis_dir.mkdir(parents=True, exist_ok=True)
                start_list = [
                    p["start"].tolist() if p is not None else None
                    for p in cost_eval_pbs
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
                )
                ax.legend()
                figname = "cost={:.2f}.jpg".format(cost_intervals[ii])
                fig.savefig(vis_dir.joinpath(figname), dpi=300)
                plt.close(fig=fig)
            cost_logs = gather_log(
                eval_stats=cost_eval_i,
                names_n_keys={
                    "attr_vals": ["cum_costs"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            cost_eval_stats[ii] = {
                "vals": cost_logs["attr_vals"],
                "pred": cost_logs["attr_pred"],
                "ref": cost_intervals[ii],
                "success": cost_logs["success_hist"],
            }
        else:
            print("[WARN] empty set for cost eval problem")

    if len(illustration_pb_file) > 0:
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
            if collect_trajs:
                assert vis_dir is not None, "vis_dir is not defined!"
                vis_dir.mkdir(parents=True, exist_ok=True)
                start_list = [
                    p["start"].tolist() if p is not None else None
                    for p in cost_eval_pbs
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
                for jj in range(len(start_list)):
                    xy_n = eval_env.normalize_obs(start_list[jj])
                    ax.text(
                        x=xy_n[0] + 0.1,
                        y=xy_n[1],
                        s="cost={:.2f}, predicted cost={:.2f}".format(
                            cost_eval_i[jj]["cum_costs"],
                            cost_eval_pbs[jj]["info"]["prediction"],  # type: ignore
                        ),
                    )
                ax.set_title("illustration problems")
                ax.legend()
                figname = "ref.jpg"
                fig.savefig(vis_dir.joinpath(figname), dpi=300)
                plt.close(fig=fig)

    eval_stats = {}
    eval_stats["dists"] = dist_eval_stats
    eval_stats["costs"] = cost_eval_stats
    return eval_stats
