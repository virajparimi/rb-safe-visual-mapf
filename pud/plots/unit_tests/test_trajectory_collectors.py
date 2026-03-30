import torch
from tqdm import tqdm
import yaml
import unittest
import numpy as np
import networkx as nx
from typing import List
from dotmap import DotMap

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.policies import SearchPolicy
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointBlendWrapper, SafeGoalConditionedPointQueueWrapper, SafeGoalConditionedPointWrapper, safe_env_load_fn
from pud.utils import set_env_seed, set_global_seed


NUM_SAMPLES = 5
DEVICE = "cuda:0"
REPLAY_BUFFER_SIZE = 1000


class TestPointEnvTrajectoryCollector(unittest.TestCase):
    def setUp(self):
        self.problem_set_file = "pud/plots/data/centerdot/easy.npz"
        self.unconstrained_ckpt_file = "models/CenterDot/ckpt/ckpt_0300000"
        self.config_file = "models/CenterDot/lag/2024-07-30-21-31-48/bk/bk_config.yaml"
        self.constrained_ckpt_file = "models/CenterDot/lag/2024-07-30-21-31-48/ckpt/ckpt_0600000"

    def pointenv_setup(self):
        assert len(self.config_file) > 0
        assert len(self.constrained_ckpt_file) > 0

        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)

        config.device = DEVICE
        config.num_samples = NUM_SAMPLES
        config.replay_buffer.max_size = REPLAY_BUFFER_SIZE

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

        obs_dim = eval_env.observation_space["observation"].shape[0]  # type: ignore
        goal_dim = obs_dim
        state_dim = obs_dim + goal_dim
        action_dim = eval_env.action_space.shape[0]  # type: ignore
        max_action = float(eval_env.action_space.high[0])  # type: ignore

        agent = DRLDDPGLag(
            state_dim,  # Concatenating obs and goal
            action_dim,
            max_action,
            CriticCls=GoalConditionedCritic,
            device=torch.device(config.device),
            **config.agent,
        )

        agent.load_state_dict(
            torch.load(self.constrained_ckpt_file, map_location=torch.device(config.device))
        )
        agent.to(torch.device(config.device))
        agent.eval()

        return config, eval_env, agent

    def load_agent_and_env(self, agent, eval_env, config, constrained=False):
        if constrained:
            agent.load_state_dict(
                torch.load(
                    self.constrained_ckpt_file, map_location=torch.device(config.device)
                )
            )
        else:
            agent.load_state_dict(
                torch.load(
                    self.unconstrained_ckpt_file, map_location=torch.device(config.device)
                )
            )
        agent.to(torch.device(config.device))
        agent.eval()

        eval_env.duration = 300  # type: ignore
        eval_env.set_use_q(True)  # type: ignore
        eval_env.set_prob_constraint(1.0)  # type: ignore

        return agent, eval_env

    def load_problem_set(self, file_path):
        load = np.load(file_path, allow_pickle=True)
        rb_vec = load["rb_vec"]
        unconstrained_pdist = load["unconstrained_pdist"]
        constrained_pdist = load["constrained_pdist"]
        pcost = load["pcost"]
        problems = load["problems"]
        return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems.tolist()

    def test_single_unconstrained_policy(self):
        config, eval_env, agent = self.pointenv_setup()
        agent, eval_env = self.load_agent_and_env(agent, eval_env, config)
        problem_setup = self.load_problem_set(self.problem_set_file)
        problems = problem_setup[-1].copy()
        eval_env.set_pbs(pb_list=problems)

        for _ in tqdm(range(config.num_samples), total=config.num_samples, desc="Problem Instance"):
            _, _, _, _, _, _ = ConstrainedCollector.get_trajectory(
                agent, eval_env, habitat=False
            )

    def test_single_unconstrained_search_policy(self):
        config, eval_env, agent = self.pointenv_setup()
        agent, eval_env = self.load_agent_and_env(agent, eval_env, config)
        problem_setup = self.load_problem_set(self.problem_set_file)
        rb_vec, pdist = problem_setup[0].copy(), problem_setup[1].copy()
        problems = problem_setup[-1].copy()
        eval_env.set_pbs(pb_list=problems)

        cbs_config = (
            {
                "seed": None,
                "max_time": 300,
                "max_distance": eval_env.max_goal_dist,
                "use_experience": True,
                "use_cardinality": True,
                "collision_radius": 0.0,
                "risk_attribute": "cost",
                "edge_attributes": ["step"],
                "split_strategy": "disjoint",
            }
        )
        search_policy = SearchPolicy(
            agent,
            pdist=pdist,
            rb_vec=rb_vec,
            open_loop=True,
            cbs_config=cbs_config,
            no_waypoint_hopping=True,
        )

        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        for _ in tqdm(range(config.num_samples)):
            _, _, _, _, _, _ = ConstrainedCollector.get_trajectory(
                    search_policy, eval_env, habitat=False
                )

    def test_single_constrained_policy(self):
        config, eval_env, agent = self.pointenv_setup()
        agent, eval_env = self.load_agent_and_env(agent, eval_env, config, constrained=True)
        problem_setup = self.load_problem_set(self.problem_set_file)
        problems = problem_setup[-1].copy()
        eval_env.set_pbs(pb_list=problems)

        for _ in tqdm(range(config.num_samples), total=config.num_samples, desc="Problem Instance"):
            _, _, _, _, _, _ = ConstrainedCollector.get_trajectory(
                agent, eval_env, habitat=False
            )

    def test_single_constrained_search_policy(self):
        config, eval_env, agent = self.pointenv_setup()
        agent, eval_env = self.load_agent_and_env(agent, eval_env, config, constrained=True)
        problem_setup = self.load_problem_set(self.problem_set_file)
        problems = problem_setup[-1].copy()
        eval_env.set_pbs(pb_list=problems)




if __name__ == "__main__":
    unittest.main()
