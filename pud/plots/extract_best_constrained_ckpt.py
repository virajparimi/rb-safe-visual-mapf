import yaml
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.vision.vision_agent import LagVisionUVFDDPG
from pud.utils import set_global_seed, set_env_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
    SafeGoalConditionedPointQueueWrapper,
)


def pointenv_setup(args):
    assert len(args.config_file) > 0
    assert len(args.constrained_ckpt_file) > 0

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # User defined parameters for evaluation
    trained_cost_limit = config.agent.cost_limit

    config.device = args.device
    config.num_samples = args.num_samples
    config.replay_buffer.max_size = args.replay_buffer_size

    set_global_seed(config.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in config.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

    eval_env = safe_env_load_fn(
        config.env.toDict(),
        config.cost_function.toDict(),
        max_episode_steps=config.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )

    set_env_seed(eval_env, config.seed + 2)

    obs_dim = eval_env.observation_space['observation'].shape[0]  # type: ignore
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    logging.debug(f'Obs dim: {obs_dim},\n'
                  f'Goal dim: {goal_dim},\n'
                  f'State dim: {state_dim},\n'
                  f'Action dim: {action_dim},\n'
                  f'Max action: {max_action}')

    agent = DRLDDPGLag(
            state_dim,  # Concatenating obs and goal
            action_dim,
            max_action,
            CriticCls=GoalConditionedCritic,
            device=torch.device(config.device),
            **config.agent,
        )

    agent.load_state_dict(torch.load(args.constrained_ckpt_file))
    agent.to(torch.device(config.device))
    agent.eval()

    return config, eval_env, agent, trained_cost_limit


def habitat_setup(args):
    assert len(args.config_file) > 0
    assert len(args.constrained_ckpt_file) > 0

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # User defined parameters for evaluation
    trained_cost_limit = config.agent_cost_kwargs.cost_limit

    config.device = args.device
    config.num_samples = args.num_samples
    config.replay_buffer.max_size = args.replay_buffer_size

    set_global_seed(config.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in config.wrappers:
        if wrapper_name == "GoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(GoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

    eval_env = safe_habitat_env_load_fn(
        env_kwargs=config.env.toDict(),
        cost_f_args=config.cost_function.toDict(),
        cost_limit=config.agent_cost_kwargs.cost_limit,
        max_episode_steps=config.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
        )
    set_env_seed(eval_env, config.seed + 1)

    config.agent["action_dim"] = eval_env.action_space.shape[0]
    config.agent["max_action"] = float(eval_env.action_space.high[0])

    agent = LagVisionUVFDDPG(
        width=config.env.simulator_settings.width,
        height=config.env.simulator_settings.height,
        in_channels=4,
        act_fn=torch.nn.SELU,
        encoder="VisualEncoder",
        device=config.device,
        **config.agent.toDict(),
        cost_kwargs=config.agent_cost_kwargs.toDict(),
    )

    agent.load_state_dict(torch.load(args.constrained_ckpt_file))
    agent.to(torch.device(config.device))
    agent.eval()

    return config, eval_env, agent, trained_cost_limit


def load_agent_and_env(agent, eval_env, args, config, constrained=False):
    if constrained:
        agent.load_state_dict(torch.load(args.constrained_ckpt_file))
    else:
        agent.load_state_dict(torch.load(args.unconstrained_ckpt_file))
    agent.to(torch.device(config.device))
    agent.eval()

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    return agent, eval_env


def load_problem_set(file_path, env, agent):
    load = np.load(file_path, allow_pickle=True)
    rb_vec = load["rb_vec"]
    unconstrained_pdist = load["unconstrained_pdist"]
    constrained_pdist = load["constrained_pdist"]
    pcost = load["pcost"]
    problems = load["problems"]
    return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems.tolist()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")

    args = parser.parse_args()
    return args


def extract_metrics(records):
    success_rate = 0.0
    steps = []
    rewards = []
    cumulative_costs = []
    for record in records:
        if record["cumulative_costs"] < trained_cost_limit:
            if record["success"]:
                success_rate += 1
        steps.append(record["steps"])
        rewards.append(record["rewards"])
        cumulative_costs.append(record["cumulative_costs"])

    metrics = {
        'steps': steps,
        'rewards': rewards,
        'cumulative_costs': cumulative_costs,
        'success_rate': success_rate / len(records),
    }
    return metrics


def single_constrained_policy(agent, eval_env, problem_setup, args, config):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=True)

    habitat = args.visual
    problems = problem_setup[-1]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained_records = []
    if args.visual:
        for tqdm_idx in tqdm(range(config.num_samples)):
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(agent, eval_env, habitat=habitat)
            constrained_records.append(records)
    else:
        for _ in range(config.num_samples):
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(agent, eval_env, habitat=habitat)
            constrained_records.append(records)

    return constrained_records


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = argument_parser()
    if args.visual:
        config, eval_env, agent, trained_cost_limit = habitat_setup(args)
    else:
        config, eval_env, agent, trained_cost_limit = pointenv_setup(args)
    assert len(args.problem_set_file) > 0
    problem_setup = load_problem_set(args.problem_set_file, eval_env, agent)

    constrained_ckpt_dir = Path(args.constrained_ckpt_file).parent
    records_dir = constrained_ckpt_dir.parent / "records"
    num_records = len(list(records_dir.glob("*.npy")))

    logging.info(f"Starting from {num_records} ckpt")

    ckpts = sorted(list(constrained_ckpt_dir.glob("ckpt_*")))
    ckpts = ckpts[num_records:]

    pbar = tqdm(ckpts)

    ckpt_file_names = []
    success_rates = []
    for ckpt in ckpts:
        args.constrained_ckpt_file = str(ckpt)
        record = single_constrained_policy(agent, eval_env, problem_setup, args, config)
        np.save(ckpt.parent.parent / "records" / f"{ckpt.stem}.npy", record)
        metric = extract_metrics(record)
        ckpt_file_names.append(ckpt)
        success_rates.append(metric["success_rate"])
        pbar.update(1)
    pbar.close()

    srates = []
    for record_file in records_dir.glob("*.npy"):
        record = np.load(record_file, allow_pickle=True)
        metric = extract_metrics(record)
        srates.append(metric["success_rate"])

    ckpt_files = sorted(list(constrained_ckpt_dir.glob("ckpt_*")))
    logging.info(f"Mean success rate: {np.mean(srates)}")
    logging.info(f"Std success rate: {np.std(srates)}")
    logging.info(f"Max success rate: {np.max(srates)}")
    logging.info(f"Argmax success rate: {np.argmax(srates)}, {ckpt_files[np.argmax(srates)]}")
    logging.info(f"Min success rate: {np.min(srates)}")
    logging.info(f"Argmin success rate: {np.argmin(srates)}, {ckpt_files[np.argmin(srates)]}")

    best_ckpt = ckpt_files[np.argmax(srates)]
    best_ckpt_file = constrained_ckpt_dir / "best_ckpt.pth"
    best_ckpt_file.write_bytes(best_ckpt.read_bytes())

    logging.info("Top 10 ckpts")
    top_10 = np.argsort(srates)[-10:]
    for idx in top_10[::-1]:
        logging.info(f"{idx + 1}: {srates[idx]}, {ckpt_files[idx]}")

    # np.save("constrained_ckpt_file_names.npy", ckpt_file_names)
    # np.save("constrained_ckpt_success_rates.npy", success_rates)
