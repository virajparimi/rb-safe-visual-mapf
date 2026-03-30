import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
import matplotlib.pyplot as plt

from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.collectors.constrained_collector import eval_agent_from_Q
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import ConstrainedCollector as Collector
from pud.envs.safe_pointenv.pb_sampler import (
    load_pb_set,
    sample_pbs_by_agent,
    sample_cost_pbs_by_agent,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)
from pud.algos.policies import (
    SearchPolicy,
    ConstrainedSearchPolicy,
    MultiAgentSearchPolicy,
    ConstrainedMultiAgentSearchPolicy,
)
from pud.visualizers.visualize import (
    visualize_graph,
    visualize_buffer,
    visualize_cost_graph,
    visualize_trajectory,
    visualize_search_path,
    visualize_eval_records,
    visualize_combined_graph,
    visualize_pairwise_costs,
    visualize_pairwise_dists,
    visualize_graph_ensemble,
    visualize_compare_search,
    visualize_combined_graph_ensemble,
)


def setup_args_parser(parser):
    parser.add_argument("--env", type=str, default="", help="Environment name")
    parser.add_argument(
        "--action_noise", type=float, default=-1, help="Action noise from environment"
    )
    parser.add_argument(
        "--resize_factor", type=int, default=-1, help="Override default resize factor"
    )
    parser.add_argument("--ckpt", type=str, help="The path to ckpt file")
    parser.add_argument(
        "--constrained_ckpt",
        type=str,
        default="",
        help="The path to constrained ckpt file",
    )
    parser.add_argument(
        "--unconstrained_ckpt",
        type=str,
        default="",
        help="The path to unconstrained ckpt file",
    )
    parser.add_argument("--figsavedir", type=str, help="Directory to save figures")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--buffer_size", type=int, default=-1, help="Replay buffer size"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on. cpu | cuda"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="Training configuration",
    )
    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        default="",
        help="Path to illustration reference problem",
    )
    parser.add_argument("--cost_name", type=str, default="", help="Override cost type")
    parser.add_argument(
        "--cost_radius", type=float, default=-1, help="Override cost type"
    )
    parser.add_argument("--cost_max", type=float, default=-1, help="Override if > 0")
    parser.add_argument("--cost_N", type=int, default=0, help="Override if > 0")
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="Override cost limit"
    )
    return parser


def setup_env(args):
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    # For dot completion
    cfg = DotMap(cfg)

    # Override cfgs from terminal
    if len(args.env) > 0:
        cfg.env.walls = args.env
    if args.resize_factor > 0:
        cfg.env.resize_factor = args.resize_factor
    if args.action_noise > 0:
        cfg.env.action_noise = args.action_noise
    if len(args.cost_name) > 0:
        cfg.cost_function.name = args.cost_name
    if args.cost_radius > 0:
        cfg.cost_function.radius = args.cost_radius
    if args.cost_max > 0:
        cfg.agent.cost_max = args.cost_max
    if args.cost_N > 0:
        cfg.agent.cost_N = args.cost_N
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    cfg.pprint()
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    if args.buffer_size > 0:
        cfg.replay_buffer.max_size = args.buffer_size

    cfg.pprint()

    figdir = Path(args.figsavedir)
    figdir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    eval_env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )

    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = eval_env.observation_space["observation"].shape[0]  # type: ignore
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    assert eval_env.action_space.shape is not None
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])  # type: ignore
    print(
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
        device=torch.device(cfg.device),
        **cfg.agent,
    )

    ckpt_file = args.constrained_ckpt
    agent.load_state_dict(torch.load(ckpt_file))
    agent.to(torch.device(args.device))
    agent.eval()

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )
    return dict(
        cfg=cfg,
        agent=agent,
        obs_dim=obs_dim,
        eval_env=eval_env,
        figsavedir=figdir,
        goal_dim=goal_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        replay_buffer=replay_buffer,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = setup_args_parser(parser)
    args = parser.parse_args()

    setup_ret = setup_env(args)

    cfg = setup_ret["cfg"]
    agent = setup_ret["agent"]
    obs_dim = setup_ret["obs_dim"]
    eval_env = setup_ret["eval_env"]
    goal_dim = setup_ret["goal_dim"]
    figdir = setup_ret["figsavedir"]
    state_dim = setup_ret["state_dim"]
    action_dim = setup_ret["action_dim"]
    max_action = setup_ret["max_action"]
    replay_buffer = setup_ret["replay_buffer"]

    assert isinstance(figdir, Path)
    # We'll give the agent lots of time to try to find the goal.
    eval_env.duration = 100  # type: ignore
    visualize_trajectory(
        agent,
        eval_env,
        difficulty=0.5,
        outpath=figdir.joinpath("vis_traj.jpg").as_posix(),
    )

    # We now will implement the search policy, which automatically finds these waypoints via graph search.
    # The first step is to fill the replay buffer with random data.

    eval_env.set_sample_goal_args(  # type: ignore
        min_dist=0,
        min_cost=0.0,
        max_cost=1.0,
        max_dist=np.inf,
        prob_constraint=0.0,
    )

    # Replay buffer is normalized between 0 and 1
    rb_vec = Collector.sample_initial_unconstrained_states(eval_env, replay_buffer.max_size)  # type: ignore

    visualize_buffer(
        rb_vec, eval_env, outpath=figdir.joinpath("vis_buffer.jpg").as_posix()
    )

    # ensemble, rb_vec, rb_vec
    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore

    agent.load_state_dict(torch.load(args.unconstrained_ckpt))  # type: ignore
    agent.to(torch.device(args.device))  # type: ignore
    agent.eval()  # type: ignore

    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore

    agent.load_state_dict(torch.load(args.constrained_ckpt))  # type: ignore
    agent.to(torch.device(args.device))  # type: ignore
    agent.eval()  # type: ignore

    visualize_cost_graph(
        pcost=pcost,
        rb_vec=rb_vec,
        eval_env=eval_env,
        edges_to_display=10,
        cost_limit=cfg.agent.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_cost_graph.jpg").as_posix(),
    )

    visualize_combined_graph(
        cutoff=7,
        pdist=pdist,
        pcost=pcost,
        rb_vec=rb_vec,
        eval_env=eval_env,
        edges_to_display=10,
        cost_limit=cfg.agent.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_combined_graph.jpg").as_posix(),
    )

    # As a sanity check, we'll plot the pairwise distances between all
    # observations in the replay buffer. We expect to see a range of values
    # from 1 to 20. Distributional RL implicitly caps the maximum predicted
    # distance by the largest bin. We've used 20 bins, so the critic
    # predicts 20 for all states that are at least 20 steps away from one another.

    visualize_pairwise_dists(pdist, outpath=figdir.joinpath("vis_pdist.jpg").as_posix())
    visualize_pairwise_costs(
        pcost,
        n_bins=cfg.agent.cost_N,  # type: ignore
        cost_limit=cfg.agent.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_pcost.jpg").as_posix(),
    )

    # With these distances, we can construct a graph. Nodes in the graph are
    # observations in our replay buffer. We connect observations with edges
    # whose lengths are equal to the predicted distance between those observations.
    # Since it is hard to visualize the edge lengths, we included a slider that
    # allows you to only show edges whose predicted length is less than some threshold.

    # Our method learns a collection of critics, each of which makes an independent
    # prediction for the distance between two states. Because each network may make
    # bad predictions for pairs of states it hasn't seen before, we act in
    # a *risk-averse* manner by using the maximum predicted distance across our
    # ensemble. That is, we act pessimistically, only adding an edge
    # if *all* critics think that this pair of states is nearby.

    visualize_graph(
        rb_vec, eval_env, pdist, outpath=figdir.joinpath("vis_graph.jpg").as_posix()
    )

    # We can also visualize the predictions from each critic.
    # Note that while each critic may make incorrect decisions
    # for distant states, their predictions in aggregate are correct.

    visualize_graph_ensemble(
        rb_vec,
        eval_env,
        pdist,
        outpath=figdir.joinpath("vis_graph_ensemble.jpg").as_posix(),
    )

    visualize_combined_graph_ensemble(
        rb_vec,
        eval_env,
        pdist,
        pcost,
        cfg.agent.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_combined_graph_ensemble.jpg").as_posix(),
    )

    # Rollout trained policy and visualize trajectory
    eval_env.set_prob_constraint(1.0)  # type: ignore
    pbs_c = sample_cost_pbs_by_agent(
        K=1,
        min_dist=10,
        max_dist=20,
        agent=agent,  # type: ignore
        env=eval_env,  # type: ignore
        num_states=50,
        target_val=0.5,
        ensemble_agg="mean",
    )
    eval_env.set_pbs(pb_list=pbs_c)  # type: ignore
    num_pb_c = len(pbs_c)

    goal_list = [p["goal"].tolist() for p in pbs_c]  # type: ignore
    start_list = [p["start"].tolist() for p in pbs_c]  # type: ignore

    eval_records = eval_agent_from_Q(
        policy=agent, eval_env=eval_env, collect_trajs=True
    )

    fig, ax = plt.subplots()
    visualize_eval_records(
        ax=ax,
        goals=goal_list,
        starts=start_list,
        eval_env=eval_env,  # type: ignore
        eval_records=eval_records,
    )
    fig.savefig(figdir.joinpath("test_trajs.jpg"), dpi=300)
    plt.close(fig)

    # Manually craft a few test cases to test cost constraint
    goal = eval_env.de_normalize_obs([0.45, 0.32])  # type: ignore
    start = eval_env.de_normalize_obs([0.32, 0.45])  # type: ignore
    pb_c_1 = {
        "start": start,
        "goal": goal,
    }
    eval_env.set_pbs(pb_list=[pb_c_1])  # type: ignore
    eval_records = eval_agent_from_Q(
        policy=agent, eval_env=eval_env, collect_trajs=True
    )

    fig, ax = plt.subplots()
    visualize_eval_records(
        ax=ax,
        goals=[goal],
        starts=[start],
        eval_env=eval_env,  # type: ignore
        eval_records=eval_records,
    )
    fig.savefig(figdir.joinpath("test_pbs.jpg"), dpi=300)
    plt.close(fig)

    goal = eval_env.de_normalize_obs([0.55, 0.68])  # type: ignore
    start = eval_env.de_normalize_obs([0.68, 0.55])  # type: ignore
    pb_c_2 = {
        "start": start,
        "goal": goal,
    }
    eval_env.set_pbs(pb_list=[pb_c_2])  # type: ignore
    eval_records = eval_agent_from_Q(
        policy=agent, eval_env=eval_env, collect_trajs=True
    )

    fig, ax = plt.subplots()
    visualize_eval_records(
        ax=ax,
        goals=[goal],
        starts=[start],
        eval_env=eval_env,  # type: ignore
        eval_records=eval_records,
    )
    fig.savefig(figdir.joinpath("test_pbs_2.jpg"), dpi=300)
    plt.close(fig)

    goal = eval_env.de_normalize_obs([0.55, 0.68])  # type: ignore
    start = eval_env.de_normalize_obs([0.68, 0.45])  # type: ignore
    pb_c_3 = {
        "start": start,
        "goal": goal,
    }
    eval_env.set_pbs(pb_list=[pb_c_3])  # type: ignore
    eval_records = eval_agent_from_Q(
        policy=agent, eval_env=eval_env, collect_trajs=True
    )

    fig, ax = plt.subplots()
    visualize_eval_records(
        ax=ax,
        goals=[goal],
        starts=[start],
        eval_env=eval_env,  # type: ignore
        eval_records=eval_records,
    )
    fig.savefig(figdir.joinpath("test_pbs_3.jpg"), dpi=300)
    plt.close(fig)

    if len(args.illustration_pb_file) > 0:
        problems = load_pb_set(file_path=args.illustration_pb_file, env=eval_env, agent=agent)  # type: ignore
    else:
        problems = sample_pbs_by_agent(
            K=10,
            min_dist=0,
            agent=agent,  # type: ignore
            env=eval_env,  # type: ignore
            target_val=10,
            num_states=100,
            ensemble_agg="mean",
            use_uncertainty=False,
            max_dist=eval_env.max_goal_dist,  # type: ignore
        )
        assert len(problems) > 0

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    eval_env.set_use_q(True)  # type: ignore

    search_policy = SearchPolicy(
        agent, rb_vec, pdist=pdist, open_loop=True, no_waypoint_hopping=True
    )
    # We'll give the agent lots of time to try to find the goal.
    eval_env.duration = 300  # type: ignore

    # Plot the search path found by the search policy
    visualize_search_path(
        search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_search.jpg").as_posix(),
    )

    # Now, we'll use that path to guide the agent towards the goal.
    # On the left, we plot rollouts from the baseline goal-conditioned policy.
    # On the right, we use that same policy to reach each of the waypoints
    # leading to the goal. As before, the slider allows you to change the
    # distance to the goal. Note that only the search policy is able to reach distant goals.

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    visualize_compare_search(
        agent,
        search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare.jpg").as_posix(),
    )

    constrained_search_policy = ConstrainedSearchPolicy(
        agent,
        rb_vec,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        no_waypoint_hopping=True,
        max_cost_limit=cfg.agent.cost_limit,  # type: ignore
        ckpts={
            "unconstrained": args.unconstrained_ckpt,
            "constrained": args.constrained_ckpt,
        },
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_search_path(
        constrained_search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_constrained_search.jpg").as_posix(),
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_compare_search(
        agent,
        constrained_search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare_constrained.jpg").as_posix(),
    )

    num_agents = 4
    ma_search_policy = MultiAgentSearchPolicy(
        agent, rb_vec, num_agents, pdist=pdist, open_loop=True, no_waypoint_hopping=True
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_search_path(
        ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_multi_agent_search.jpg").as_posix(),
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_compare_search(
        agent,
        ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare_multi_agent.jpg").as_posix(),
    )

    constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
        agent,
        rb_vec,
        num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        no_waypoint_hopping=True,
        max_cost_limit=cfg.agent.cost_limit,  # type: ignore
        ckpts={
            "unconstrained": args.unconstrained_ckpt,
            "constrained": args.constrained_ckpt,
        },
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_search_path(
        constrained_ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_constrained_multi_agent_search.jpg").as_posix(),
    )

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    visualize_compare_search(
        agent,
        constrained_ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare_constrained_multi_agent.jpg").as_posix(),
    )
