import yaml
import torch
import argparse
import numpy as np
from dotmap import DotMap
from gym.spaces import Box

from pud.algos.policies import GaussianPolicy
from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.pb_sampler import sample_pbs_by_agent
from pud.collectors.constrained_collector import eval_agent_from_Q
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointQueueWrapper,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_PointEnv_Queue.yaml",
        help="Training configuration",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda")

    args = parser.parse_args()
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)

    cfg.device = args.device
    cfg.pprint()

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointQueueWrapper":
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
    assert isinstance(eval_env.action_space, Box)
    max_action = float(eval_env.action_space.high[0])
    print(
        f"Observation dimension: {obs_dim},\n"
        f"Goal dimension: {goal_dim},\n"
        f"State dimension: {state_dim},\n"
        f"Action dimension: {action_dim},\n"
        f"Max Action: {max_action}"
    )

    agent = DRLDDPGLag(
        # DDPG args
        state_dim,  # Concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
    )
    agent.to(torch.device(args.device))

    policy = GaussianPolicy(agent)
    print(agent)

    assert isinstance(eval_env, SafeGoalConditionedPointQueueWrapper)
    eval_env.set_use_q(True)
    eval_env.set_verbose(True)

    # Test one special problem
    pb_list = [
        {
            "start": np.array([0, 0]),
            "goal": np.array([0, 0]),
            "info": {"prediction": 1.0},
        },
    ]
    eval_env.set_pbs(pb_list=[(pb["start"], pb["goal"], pb["info"]) for pb in pb_list])
    target_init_states = {
        "observation": eval_env.normalize_obs(pb_list[0]["start"]),
        "goal": eval_env.normalize_obs(pb_list[0]["goal"]),
    }
    assert eval_env.get_Q_size() == 1
    eval_env.set_prob_constraint(1.0)
    eval_stats = eval_agent_from_Q(policy=agent, eval_env=eval_env)
    assert eval_env.get_Q_size() == 0

    pbs = sample_pbs_by_agent(
        K=5,
        agent=agent,
        env=eval_env,
        num_states=100,
        target_val=1.0,
        ensemble_agg="max",
    )
    eval_env.append_pbs([(pb["start"], pb["goal"], pb["info"]) for pb in pbs])
    eval_stats = eval_agent_from_Q(policy=agent, eval_env=eval_env)

    for i in range(len(pbs)):
        np.allclose(
            eval_stats[i]["init_states"]["observation"],
            eval_env.normalize_obs(pbs[i]["start"]),
        )
        np.allclose(
            eval_stats[i]["init_states"]["goal"],
            eval_env.normalize_obs(pbs[i]["goal"]),
        )

    # Logging
    attr = "cum_costs"
    attr_vals = []
    attr_pred = []
    for id in eval_stats:
        attr_vals.append(eval_stats[id][attr])
        attr_pred.append(eval_stats[id]["init_info"]["prediction"])
