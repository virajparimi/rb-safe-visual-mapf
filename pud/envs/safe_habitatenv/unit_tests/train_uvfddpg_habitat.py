import yaml
import torch
import argparse
from pathlib import Path
from dotmap import DotMap
from gym.spaces import Box
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

from pud.algos.policies import GaussianPolicy
from pud.collectors.collector import Collector
from pud.utils import set_env_seed, set_global_seed
from pud.algos.vision.vision_agent import VisionUVFDDPG
from pud.buffers.visual_buffer import VisualReplayBuffer
from pud.runners.runner import train_eval, eval_pointenv_dists
from pud.envs.habitat_navigation_env import (
    GoalConditionedHabitatPointWrapper,
    habitat_env_load_fn,
)

from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafeHabitatEnv.yaml",
        help="Training configuration",
    )
    parser.add_argument("--cost_name", type=str, default="", help="override cost type")
    parser.add_argument(
        "--cost_radius", type=float, default=-1, help="override cost type"
    )
    parser.add_argument("--cost_max", type=float, default=-1, help="override if > 0")
    parser.add_argument("--cost_N", type=int, default=0, help="override if > 0")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="if non-empty, load the checkpoint into agent model",
    )
    parser.add_argument(
        "--i_start",
        type=int,
        default=1,
        help="override the start iteration index, for resume training",
    )
    parser.add_argument(
        "--collect_steps",
        type=int,
        default=-1,
        help="override the number of steps per agent update",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=-1, help="override num of iterations"
    )
    parser.add_argument(
        "--cost_limit", type=float, default=-1, help="override cost limit"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        help="override scene",
    )
    parser.add_argument(
        "--lambda_lr", type=float, default=-1, help="override lagrange lr"
    )
    parser.add_argument(
        "--apsp_path", type=str, default="", help="supply pre-computed apsp path"
    )
    parser.add_argument(
        "--illustration_pb_file",
        type=str,
        help="problems that serve as illustration and evaluation set",
    )
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--visual", action="store_true", help="generate and save visual trajs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)

    # Override cfs from terminal
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
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    if len(args.apsp_path) > 0:
        cfg.env.apsp_path = args.apsp_path
    if len(args.scene) > 0:
        cfg.env.scene = args.scene
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    cfg.pprint()

    # Custom Logging
    log_dir = Path(cfg.ckpt_dir)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = log_dir.joinpath(date_time)
    ckpt_dir = log_dir.joinpath("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tfevent_dir = log_dir.joinpath("tfevent")
    tfevent_dir.mkdir(parents=True, exist_ok=True)
    bk_dir = log_dir.joinpath("bk")
    bk_dir.mkdir(parents=True, exist_ok=True)
    with open(bk_dir.joinpath("config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    tb = SummaryWriter(log_dir=tfevent_dir.as_posix())

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

    cfg.env.device = args.device

    env = habitat_env_load_fn(
        env_type=cfg.env.env_type,
        height=cfg.env.height,
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        apsp_path=cfg.env.apsp_path,
        simulator_settings=cfg.env.simulator_settings,
        device=cfg.device,
        terminate_on_timeout=False,
    )
    set_env_seed(env, cfg.seed + 1)

    eval_env = habitat_env_load_fn(
        env_type=cfg.env.scene,
        height=cfg.env.height,
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        apsp_path=cfg.env.apsp_path,
        simulator_settings=cfg.env.simulator_settings,
        device=cfg.device,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + 2)

    uvfddpg_kwargs = cfg.agent.toDict()
    uvfddpg_kwargs["state_dim"] = 256 * 2  # Latent state dim
    assert env.action_space.shape is not None
    uvfddpg_kwargs["action_dim"] = env.action_space.shape[0]
    assert isinstance(env.action_space, Box)
    uvfddpg_kwargs["max_action"] = float(env.action_space.high[0])

    agent = VisionUVFDDPG(
        in_channels=4,
        embedding_size=256,
        act_fn=torch.nn.SELU,
        device=cfg.device,
        width=cfg.env.simulator_settings.width,
        height=cfg.env.simulator_settings.height,
        encoder=args.encoder,
        **cfg.agent.toDict(),
    )
    agent.to(torch.device(cfg.device))

    # Test collector
    replay_buffer = VisualReplayBuffer(
        obs_dim=(4, 64, 64, 4),
        goal_dim=(4, 64, 64, 4),
        action_dim=env.action_space.shape[0],
        max_size=1000,
    )

    policy = GaussianPolicy(agent)

    collector = Collector(
        policy,
        replay_buffer,
        env,
        initial_collect_steps=1000,
    )
    collector.step(collector.initial_collect_steps)

    train_eval(
        policy,
        agent,
        replay_buffer,
        env,
        eval_env,
        eval_func=eval_pointenv_dists,
        tensorboard_writer=tb,
        **cfg.runner,
    )
    torch.save(agent.state_dict(), ckpt_dir.joinpath("agent.pth"))
