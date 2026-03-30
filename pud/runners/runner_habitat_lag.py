import time
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from termcolor import cprint
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Union

from pud.algos.data_struct import gather_log
from pud.algos.policies import GaussianPolicy
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.envs.habitat_navigation_env import plot_wall, plot_traj
from pud.buffers.visual_buffer import ConstrainedVisualReplayBuffer
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointQueueWrapper,
)
from pud.collectors.visual_collector import (
    eval_agent_from_Q,
    ConstrainedVisualCollector,
)
from pud.envs.safe_pointenv.pb_sampler import (
    load_pb_set,
    sample_pbs_by_agent,
    sample_cost_pbs_by_agent,
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


def visualize_visual_eval_records(
    eval_records,
    eval_env,
    ax: Axes,
    starts=[],
    goals=[],
    normalize_map=False,
    color: Optional[list] = None,
    append_label: str = "",
):
    """
    For habitat env
    """

    list_trajs = []
    for id in eval_records.keys():
        list_trajs.append(eval_records[id]["traj"])

    starts = np.stack(starts)
    goals = np.stack(goals)

    ax = plot_wall(eval_env.walls.copy(), ax, normalize=normalize_map)

    for i in eval_records.keys():
        traj_color = color if color else distinct_colors[i]

        ax = plot_traj(
            traj=np.stack(eval_records[i]["traj"]),
            walls=eval_env.walls.copy(),
            normalize=normalize_map,
            ax=ax,
            color=traj_color,
            label=append_label + "traj{:0>2d}".format(i),
            marker="o",
            markersize=4,
        )

        ax.plot(
            [starts[i, 0], goals[i, 0]],
            [starts[i, 1], goals[i, 1]],
            marker="o",
            color="b",
            linestyle="--",
            linewidth=2,
            markersize=2,
            label="",
            alpha=0.3,
        )

        ax.scatter(
            goals[i : i + 1, 0],  # noqa
            goals[i : i + 1, 1],  # noqa
            marker="*",
            s=30,
            color=traj_color,
        )

    return ax


def update_train_pbs_by_metric(
    agent: LagVisionUVFDDPG,
    env: SafeGoalConditionedHabitatPointQueueWrapper,
    use_uncertainty: bool = True,  # boost samples of high uncertainty
    logger: Optional[dict] = None,
):
    # Update pbs in the train eval
    new_train_pbs = []
    if logger is None:
        logger = {}
    cost_eval_pbs = sample_cost_pbs_by_agent(
        env=env, agent=agent, use_uncertainty=use_uncertainty, **logger["sampler"]
    )
    new_train_pbs.extend(cost_eval_pbs)
    env.set_pbs(pb_list=new_train_pbs)


def train_eval(
    policy: GaussianPolicy,
    agent: LagVisionUVFDDPG,
    agent_g: LagVisionUVFDDPG,
    replay_buffer: ConstrainedVisualReplayBuffer,
    env: SafeGoalConditionedHabitatPointQueueWrapper,
    eval_env: SafeGoalConditionedHabitatPointQueueWrapper,
    num_iterations=int(1e6),
    initial_collect_steps: int = 1000,
    collect_steps: int = 1,
    opt_steps: int = 1,
    batch_size_opt: int = 64,
    eval_func=lambda agent, eval_env: None,
    opt_log_interval: int = 100,
    eval_interval: int = 10000,
    verbose: bool = True,
    pbar: bool = False,
    logger: dict = {},
):
    env.set_verbose(False)
    env.set_use_q(True)
    # Train cost critic but not penalize unsafe actions
    agent.set_lag_status(turn_on_lag=True)

    time_logs = log_time(step=0)
    collector = ConstrainedVisualCollector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )

    ep_cost = 0.0
    num_eps = collector.num_eps

    collector.step(collector.initial_collect_steps)

    for i in tqdm(
        range(1, num_iterations + 1),
        total=num_iterations,
        disable=not pbar,
        desc="training",
    ):

        logger["i"] = i
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
            if "ckpt" in logger and isinstance(logger["ckpt"], Path):
                torch.save(
                    agent.state_dict(),
                    logger["ckpt"].joinpath("ckpt_{:0>7d}".format(i)),
                )

            agent.eval()
            if verbose:
                print(f"evaluating iteration = {i}")

            # Use reference agent to generate problem samples
            update_train_pbs_by_metric(
                agent=agent_g,
                env=env,
                use_uncertainty=True,
                logger=logger,
            )

            eval_info = eval_func(
                agent=agent,
                eval_env=eval_env,
                agent_g=agent_g,
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
                logger["tb"].add_scalar(
                    "Opt/cost_critic_loss",
                    np.mean(opt_info["cost_critic_loss"]),
                    global_step=i,
                )
                logger["tb"].add_scalar(
                    "Opt/Lagrange_Multiplier",
                    agent.lagrange.lagrangian_multiplier.item(),
                    global_step=i,
                )

            if i > 1 and i % eval_interval == 0:
                time_logs = log_time(step=i, log=time_logs)
                logger["tb"].add_scalar(
                    "Time/Iters per Seconds", time_logs["speed"][-1], global_step=i
                )
                logger["tb"].add_scalar(
                    "Time/Total Time", time_logs["time"][-1], global_step=i
                )

                # For dists
                field_header = "Eval Dist ~ "
                for ii in eval_info["dists"]:
                    logger["tb"].add_scalars(
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

                    logger["tb"].add_scalars(
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
                        logger["tb"].add_scalar(
                            field_header
                            + "{:0>2d}/success_rate".format(
                                eval_info["dists"][ii]["agent"]["ref"]
                            ),
                            success_rate,
                            global_step=i,
                        )

                field_header = "Eval Cost ~ "
                for ii in eval_info["costs"]:
                    logger["tb"].add_scalars(
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
                    logger["tb"].add_scalars(
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
                        logger["tb"].add_scalar(
                            field_header + "{}/success_rate".format(ii),
                            success_rate,
                            global_step=i,
                        )
                    logger["tb"].add_scalar(
                        field_header + "{}/N".format(ii), len(N_success), global_step=i
                    )

                field_header = "Eval Ref ~ "
                for ii in eval_info["ref"]:
                    logger["tb"].add_scalars(
                        field_header + "{}/mean".format(ii),
                        tag_scalar_dict={
                            "pred": np.mean(eval_info["ref"][ii]["agent"]["pred"]),
                            "val": np.mean(eval_info["ref"][ii]["agent"]["vals"]),
                            "pred_greedy": np.mean(
                                eval_info["ref"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": np.mean(
                                eval_info["ref"][ii]["agent_g"]["vals"]
                            ),
                        },
                        global_step=i,
                    )
                    logger["tb"].add_scalars(
                        field_header + "{}/std".format(ii),
                        tag_scalar_dict={
                            "pred": np.std(eval_info["ref"][ii]["agent"]["pred"]),
                            "val": np.std(eval_info["ref"][ii]["agent"]["vals"]),
                            "pred_greedy": np.std(
                                eval_info["ref"][ii]["agent_g"]["pred"]
                            ),
                            "val_greedy": np.std(
                                eval_info["ref"][ii]["agent_g"]["vals"]
                            ),
                        },
                        global_step=i,
                    )

                    N_success = np.array(
                        eval_info["ref"][ii]["agent"]["success"], dtype=float
                    )
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        logger["tb"].add_scalar(
                            field_header + "{}/success_rate".format(ii),
                            success_rate,
                            global_step=i,
                        )
                    logger["tb"].add_scalar(
                        field_header + "{}/N".format(ii), len(N_success), global_step=i
                    )


def eval_pointenv_dists(
    agent,
    eval_env: SafeGoalConditionedHabitatPointQueueWrapper,
    agent_g: Optional[LagVisionUVFDDPG] = None,
    num_evals=10,
    sample_size: int = 100,
    eval_distances=[1, 2, 3, 4],
    cost_intervals=[0, 5, 10],
    cost_min_dist: Optional[float] = None,
    cost_max_dist: Optional[float] = None,
    verbose=True,
    logger: dict = {},
):
    if "eval_distances" in logger:
        eval_distances = logger["eval_distances"]

    eval_img_dir: Union[Path, None] = None
    if "imgs" in logger:
        eval_img_dir = logger["imgs"].joinpath("eval_{:0>5d}".format(logger["i"]))
        assert eval_img_dir is not None
        eval_img_dir.mkdir(exist_ok=True, parents=True)

    dist_eval_stats = dict()

    pbar = tqdm(
        total=len(eval_distances),
        desc="evaluating reward critic",
        disable=not logger["pbar"],
    )
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
            eval_env.append_pbs(pb_list=deepcopy(pbs))  # type: ignore
            dist_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )
            eval_env.append_pbs(pb_list=deepcopy(pbs))  # type: ignore
            dist_eval_i_g = eval_agent_from_Q(
                policy=agent_g,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )

            if eval_img_dir is not None:
                fig, ax = plt.subplots()
                start_list = [p["start"].tolist() for p in pbs]
                goal_list = [p["goal"].tolist() for p in pbs]
                visualize_visual_eval_records(
                    eval_records=dist_eval_i,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    append_label="Lag_",
                )
                visualize_visual_eval_records(
                    eval_records=dist_eval_i_g,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    color=["r"],
                )

                ax.set_title("target dist ~ {}".format(eval_distances[ii_d]))
                figname = "dist={:0>2d}.jpg".format(eval_distances[ii_d])

                fig.savefig(
                    eval_img_dir.joinpath("dist~{}.jpg".format(eval_distances[ii_d])),
                    dpi=300,
                )

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

        pbar.update()
    pbar.close()

    pbar = tqdm(
        total=len(cost_intervals),
        desc="evaluating cost critic",
        disable=not logger["pbar"],
    )
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
            eval_env.append_pbs(pb_list=deepcopy(cost_eval_pbs))  # type: ignore
            cost_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )
            eval_env.append_pbs(pb_list=deepcopy(cost_eval_pbs))  # type: ignore
            cost_eval_i_g = eval_agent_from_Q(
                policy=agent_g,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )

            if eval_img_dir is not None:
                fig, ax = plt.subplots()
                start_list = [
                    p["start"].tolist() if p is not None else None
                    for p in cost_eval_pbs
                ]
                goal_list = [
                    p["goal"].tolist() if p is not None else None for p in cost_eval_pbs
                ]
                visualize_visual_eval_records(
                    eval_records=cost_eval_i,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    append_label="Lag_",
                )
                visualize_visual_eval_records(
                    eval_records=cost_eval_i_g,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                )

                fig.savefig(
                    eval_img_dir.joinpath("cost~{}.jpg".format(cost_intervals[ii])),
                    dpi=300,
                )

                plt.close(fig=fig)

            cost_eval_stats[ii] = {}
            cost_logs = gather_log(
                eval_stats=cost_eval_i,
                names_n_keys={
                    "attr_vals": ["cum_costs"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            cost_eval_stats[ii]["agent"] = {
                "vals": cost_logs["attr_vals"],
                "pred": cost_logs["attr_pred"],
                "ref": cost_intervals[ii],
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
            cost_eval_stats[ii]["agent_g"] = {
                "vals": cost_logs_g["attr_vals"],
                "pred": cost_logs_g["attr_pred"],
                "ref": cost_intervals[ii],
                "success": cost_logs_g["success_hist"],
            }
        else:
            print("[WARN] empty set for dist eval problem")

        pbar.update()
    pbar.close()

    if "illustration_pb_file" in logger:
        ref_eval_stats = dict()
        ref_pbs = load_pb_set(
            file_path=logger["illustration_pb_file"],
            env=eval_env,
            agent=agent,
        )
        if len(ref_pbs) > 0:
            eval_env.append_pbs(pb_list=deepcopy(ref_pbs))  # type: ignore
            ref_eval_i = eval_agent_from_Q(
                policy=agent,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )
            eval_env.append_pbs(pb_list=deepcopy(ref_pbs))  # type: ignore
            ref_eval_i_g = eval_agent_from_Q(
                policy=agent_g,
                eval_env=eval_env,
                collect_trajs=not (eval_img_dir is None),
            )
            if eval_img_dir is not None:
                start_list = [
                    p["start"].tolist() if p is not None else None for p in ref_pbs
                ]
                goal_list = [
                    p["goal"].tolist() if p is not None else None for p in ref_pbs
                ]
                fig, ax = plt.subplots()
                ax = visualize_visual_eval_records(
                    eval_records=ref_eval_i,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                    append_label="Lag_",
                )
                ax = visualize_visual_eval_records(
                    eval_records=ref_eval_i_g,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                )
                text_offset = eval_env.walls.shape[0] * 0.05
                line_i = 0
                for jj in range(len(start_list)):
                    xy_n = start_list[jj]
                    assert xy_n is not None
                    ax.text(x=xy_n[0] + text_offset, y=xy_n[1], s="{}".format(jj))
                    ax.text(
                        x=0.0,
                        y=-4.0 - text_offset * line_i,
                        s="Lag traj {}: cost={:.2f}, predicted cost={:.2f}".format(
                            jj,
                            ref_eval_i[jj]["cum_costs"],
                            ref_pbs[jj]["info"]["prediction"],  # type: ignore
                        ),
                    )
                    line_i += 1
                    ax.text(
                        x=0.0,
                        y=-4.0 - text_offset * line_i,
                        s="traj {}: cost={:.2f}, predicted cost={:.2f}".format(
                            jj,
                            ref_eval_i_g[jj]["cum_costs"],
                            ref_pbs[jj]["info"]["prediction"],  # type: ignore
                        ),
                    )
                    line_i += 1

                ax.set_title("illustration problems")
                ax.legend(bbox_to_anchor=(1.01, 1.01))
                figname = "ref.jpg"
                fig.savefig(
                    eval_img_dir.joinpath(figname), dpi=300, bbox_inches="tight"
                )
                plt.close()

            ref_eval_stats[0] = {}
            ref_logs = gather_log(
                eval_stats=ref_eval_i,
                names_n_keys={
                    "attr_vals": ["cum_costs"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            ref_eval_stats[0]["agent"] = {
                "vals": ref_logs["attr_vals"],
                "pred": ref_logs["attr_pred"],
                "success": ref_logs["success_hist"],
            }
            ref_logs_g = gather_log(
                eval_stats=ref_eval_i_g,
                names_n_keys={
                    "attr_vals": ["cum_costs"],
                    "attr_pred": ["init_info", "prediction"],
                    "success_hist": ["success"],
                },
            )
            ref_eval_stats[0]["agent_g"] = {
                "vals": ref_logs_g["attr_vals"],
                "pred": ref_logs_g["attr_pred"],
                "success": ref_logs_g["success_hist"],
            }

    eval_stats = {}
    eval_stats["dists"] = dist_eval_stats
    eval_stats["costs"] = cost_eval_stats
    eval_stats["ref"] = ref_eval_stats

    return eval_stats
