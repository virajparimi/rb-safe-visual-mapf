import yaml
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import VectorGaussianPolicy
from pud.utils import set_env_seed, set_global_seed
from pud.buffers.visual_buffer import VisualReplayBuffer
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_habitatenv.unit_tests.train_uvddpg_vec_habitat import setup_logger
from pud.envs.safe_pointenv.pb_sampler import (
    load_pb_set,
    sample_pbs_by_agent,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
    safe_habitat_env_load_fn,
)
from pud.algos.policies import (
    VisualSearchPolicy,
    VisualConstrainedSearchPolicy,
    VisualMultiAgentSearchPolicy,
    VisualConstrainedMultiAgentSearchPolicy,
)
from pud.visualizers.visualize import (
    visualize_cost_graph,
    visualize_combined_graph,
    visualize_pairwise_dists,
    visualize_pairwise_costs,
    visualize_combined_graph_ensemble,
)
from pud.visualizers.visualize_habitat import (
    visualize_graph,
    visualize_buffer,
    visualize_trajectory,
    visualize_search_path,
    visualize_compare_search,
    visualize_graph_ensemble,
)


def setup_args_parser(parser):
    parser.add_argument("--scene", type=str, default="", help="Override scene")
    parser.add_argument("--cost_N", type=int, default=0, help="Override if > 0")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument("--resume", type=str, default="", help="Resume training")
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--figsavedir", type=str, help="Directory to save figures")
    parser.add_argument("--cost_max", type=float, default=-1, help="Override if > 0")
    parser.add_argument("--cost_name", type=str, default="", help="Override cost type")
    parser.add_argument(
        "--cost_radius", type=float, default=-1, help="Override cost type"
    )
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="Override cost limit"
    )
    parser.add_argument(
        "--lambda_lr", type=float, default=-1, help="Override lagrange lr"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=-1, help="Evaluation interval"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=-1, help="Learning rate for actor"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=-1, help="Learning rate for critic"
    )
    parser.add_argument(
        "--visual", action="store_true", help="Generate and save visual trajs"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=-1, help="Size of the embeddings"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on. cpu | cuda"
    )
    parser.add_argument(
        "--apsp_path", type=str, default="", help="Supply pre-computed apsp path"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=-1, help="Override num of iterations"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of envs for batch inference"
    )
    parser.add_argument(
        "--replay_buffer_size", type=int, default=-1, help="Maximum replay buffer size"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="VisualEncoder",
        help="VisualRGBEncoder | VisualEncoder",
    )
    parser.add_argument(
        "--constrained_ckpt",
        type=str,
        default="",
        help="If non-empty, load the checkpoint into agent model",
    )
    parser.add_argument(
        "--unconstrained_ckpt",
        type=str,
        default="",
        help="If non-empty, load the checkpoint into agent model",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_HabitatReplicaCAD.yaml",
        help="Training configuration",
    )
    parser.add_argument(
        "--i_start",
        type=int,
        default=1,
        help="Override the start iteration index, for resume training",
    )
    parser.add_argument(
        "--collect_steps",
        type=int,
        default=-1,
        help="Override the number of steps per agent update",
    )
    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        help="Problems that serve as illustration and evaluation set",
    )
    return parser


def override_cfgs(args, cfg):
    if len(args.logdir) > 0:
        cfg.ckpt_dir = args.logdir
    if len(args.cost_name) > 0:
        cfg.cost_function.name = args.cost_name
    if args.cost_radius > 0:
        cfg.cost_function.radius = args.cost_radius
    if args.cost_max > 0:
        cfg.agent.cost_max = args.cost_max
    if args.cost_N > 0:
        cfg.agent.cost_N = args.cost_N
    if args.lambda_lr > 0:
        cfg.agent.lambda_lr = args.lambda_lr
    if args.actor_lr > 0:
        cfg.agent.actor_lr = args.actor_lr
    if args.critic_lr > 0:
        cfg.agent.critic_lr = args.critic_lr
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    if args.eval_interval > 0:
        cfg.runner.eval_interval = args.eval_interval
    if args.replay_buffer_size > 0:
        cfg.replay_buffer.max_size = args.replay_buffer_size
    if len(args.apsp_path) > 0:
        cfg.env.apsp_path = args.apsp_path
    if len(args.scene) > 0:
        cfg.env.scene = args.scene
    if args.embedding_size > 0:
        cfg.agent.embedding_size = args.embedding_size
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    return cfg


def setup_env(args):

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = DotMap(cfg)

    # Override cfgs from terminal
    cfg = override_cfgs(args, cfg)
    cfg.pprint()

    figdir = Path(args.figsavedir)
    figdir.mkdir(parents=True, exist_ok=True)

    # Custom Logging
    logger = setup_logger(
        root_dir=cfg.ckpt_dir,
        subdir_names=["ckpt", "tfevent", "bk", "imgs"],
        tag_time=True,
    )
    with open(logger["bk"].joinpath("config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    logger["tb"] = SummaryWriter(log_dir=logger["tfevent"].as_posix())  # type: ignore

    shutil.copy("launch_jobs/cloud/cloud_debug_vec_habitat.sh", logger["bk"].as_posix())
    shutil.copy(
        "pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py",
        logger["bk"].as_posix(),
    )
    shutil.copy("pud/runners/runner_vec.py", logger["bk"].as_posix())
    shutil.copy("pud/algos/vision/visual_models.py", logger["bk"].as_posix())
    shutil.copy("pud/algos/vision/vision_agent.py", logger["bk"].as_posix())

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "GoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(GoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    eval_env = safe_habitat_env_load_fn(
        env_kwargs=cfg.env.toDict(),
        cost_f_args=cfg.cost_function.toDict(),
        cost_limit=cfg.agent_cost_kwargs.cost_limit,
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + 1)

    cfg.agent["action_dim"] = eval_env.action_space.shape[0]  # type: ignore
    cfg.agent["max_action"] = float(eval_env.action_space.high[0])  # type: ignore

    agent = LagVisionUVFDDPG(
        width=cfg.env.simulator_settings.width,
        height=cfg.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder=args.encoder,
        device=cfg.device,
        **cfg.agent.toDict(),
        cost_kwargs=cfg.agent_cost_kwargs.toDict(),
    )

    if len(args.resume) > 0:
        state_dict = torch.load(args.resume)
        agent.load_state_dict(state_dict)

    ckpt_file = args.constrained_ckpt
    agent.load_state_dict(torch.load(ckpt_file, map_location=cfg.device))
    agent.to(device=cfg.device)
    agent.eval()

    replay_buffer = VisualReplayBuffer(
        obs_dim=(
            4,
            cfg.env.simulator_settings.width,
            cfg.env.simulator_settings.height,
            4,
        ),
        goal_dim=(
            4,
            cfg.env.simulator_settings.width,
            cfg.env.simulator_settings.height,
            4,
        ),
        action_dim=eval_env.action_space.shape[0],  # type: ignore
        max_size=cfg.replay_buffer.max_size,
    )

    policy = VectorGaussianPolicy(agent, noise_scale=0.2)

    return dict(
        cfg=cfg,
        agent=agent,
        policy=policy,
        eval_env=eval_env,
        figsavedir=figdir,
        replay_buffer=replay_buffer,
    )


def sample_initial_states(eval_env, num_states):
    safe = "safe" in type(eval_env.env).__name__.lower()  # type: ignore
    rb_vec = []
    for _ in range(num_states):
        state = eval_env.reset()
        state = state[0] if safe else state
        rb_vec.append(state)
    rb_vec_grid = np.array([x["grid"]["observation"] for x in rb_vec])
    rb_vec_visual = np.array([x["observation"] for x in rb_vec])
    return rb_vec_grid, rb_vec_visual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_args_parser(parser)
    args = parser.parse_args()

    setup = setup_env(args)

    cfg = setup["cfg"]
    agent = setup["agent"]
    policy = setup["policy"]
    eval_env = setup["eval_env"]
    figdir = setup["figsavedir"]
    replay_buffer = setup["replay_buffer"]

    assert isinstance(figdir, Path)

    # We'll give the agent lots of time to try to find the goal.
    eval_env.duration = 100  # type: ignore
    visualize_trajectory(
        agent,
        eval_env,
        difficulty=0.5,
        outpath=figdir.joinpath("vis_traj.jpg").as_posix(),
    )

    rb_vec_grid, rb_vec_visual = sample_initial_states(eval_env, replay_buffer.max_size)  # type: ignore

    visualize_buffer(
        rb_vec_grid, eval_env, outpath=figdir.joinpath("vis_buffer.jpg").as_posix()
    )

    pdist = agent.get_pairwise_dist(rb_vec_visual, aggregate=None)  # type: ignore
    pcost = agent.get_pairwise_cost(rb_vec_visual, aggregate=None)  # type: ignore

    visualize_cost_graph(
        pcost=pcost,
        rb_vec=rb_vec_grid,
        eval_env=eval_env,
        edges_to_display=10,
        cost_limit=cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_cost_graph.jpg").as_posix(),
    )

    visualize_combined_graph(
        cutoff=7,
        pdist=pdist,
        pcost=pcost,
        eval_env=eval_env,
        rb_vec=rb_vec_grid,
        edges_to_display=10,
        cost_limit=cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_combined_graph.jpg").as_posix(),
    )

    visualize_pairwise_dists(pdist, outpath=figdir.joinpath("vis_pdist.jpg").as_posix())  # type: ignore
    visualize_pairwise_costs(
        pcost,
        n_bins=cfg.agent_cost_kwargs.cost_N,  # type: ignore
        cost_limit=cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_pcost.jpg").as_posix(),
    )

    visualize_graph(
        rb_vec_grid,
        eval_env,
        pdist,
        outpath=figdir.joinpath("vis_graph.jpg").as_posix(),
    )

    visualize_graph_ensemble(
        rb_vec_grid,
        eval_env,
        pdist,
        outpath=figdir.joinpath("vis_graph_ensemble.jpg").as_posix(),
    )

    visualize_combined_graph_ensemble(
        rb_vec_grid,
        eval_env,
        pdist,
        pcost,
        cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        outpath=figdir.joinpath("vis_combined_graph_ensemble.jpg").as_posix(),
    )

    if "safe" in type(eval_env.env).__name__.lower():  # type: ignore
        eval_env.set_prob_constraint(1.0)  # type: ignore

    queue_env = False
    if "queue" in type(eval_env.env).__name__.lower():  # type: ignore
        queue_env = True

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

    search_policy = VisualSearchPolicy(
        agent,
        (rb_vec_grid, rb_vec_visual),
        pdist=pdist,
        open_loop=True,
        max_search_steps=3,
        no_waypoint_hopping=True,
    )
    eval_env.duration = 300  # type: ignore

    visualize_search_path(
        search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_search.jpg").as_posix(),
    )

    eval_env.set_pbs(pb_list=problems.copy()) if queue_env else None  # type: ignore

    visualize_compare_search(
        agent,
        search_policy,
        eval_env,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare.jpg").as_posix(),  # type: ignore
    )

    constrained_search_policy = VisualConstrainedSearchPolicy(
        agent,
        (rb_vec_grid, rb_vec_visual),
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        no_waypoint_hopping=True,
        max_cost_limit=cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        ckpts={"unconstrained": args.unconstrained_ckpt, "constrained": args.constrained_ckpt},
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
    ma_search_policy = VisualMultiAgentSearchPolicy(
        agent,
        (rb_vec_grid, rb_vec_visual),
        num_agents,
        pdist=pdist,
        open_loop=True,
        max_search_steps=3,
        no_waypoint_hopping=True,
    )

    eval_env.set_pbs(pb_list=problems.copy()) if queue_env else None  # type: ignore
    visualize_search_path(
        ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_multi_agent_search.jpg").as_posix(),
    )

    eval_env.set_pbs(pb_list=problems.copy()) if queue_env else None  # type: ignore
    visualize_compare_search(
        agent,
        ma_search_policy,
        eval_env,
        num_agents=4,
        difficulty=0.9,
        outpath=figdir.joinpath("vis_compare_multi_agent.jpg").as_posix(),  # type: ignore
    )

    constrained_ma_search_policy = VisualConstrainedMultiAgentSearchPolicy(
        agent,
        (rb_vec_grid, rb_vec_visual),
        num_agents,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        no_waypoint_hopping=True,
        max_cost_limit=cfg.agent_cost_kwargs.cost_limit,  # type: ignore
        ckpts={"unconstrained": args.unconstrained_ckpt, "constrained": args.constrained_ckpt},
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
