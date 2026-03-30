import unittest
from gym.spaces import Box
from termcolor import cprint

from pud.algos.policies import GaussianPolicy
from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
)

"""
python pud/algos/unit_tests/test_collector.py TestConstrainedCollector.test_eval_agent_n_record_init_states
"""


class TestConstrainedCollector(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "walls": "CentralObstacle",
            "resize_factor": 5,
            "thin": False,
            "cost_limit": 1,
        }
        self.cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.0,
        }

        self.env = safe_env_load_fn(
            env_kwargs=self.env_kwargs,
            cost_f_kwargs=self.cost_f_kwargs,
            max_episode_steps=10,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            terminate_on_timeout=False,
        )

        assert isinstance(self.env.observation_space, dict)
        obs_dim = self.env.observation_space["observation"].shape[0]
        goal_dim = obs_dim
        state_dim = obs_dim + goal_dim
        assert self.env.action_space.shape is not None
        action_dim = self.env.action_space.shape[0]
        assert isinstance(self.env.action_space, Box)
        max_action = float(self.env.action_space.high[0])

        agent_cfg = dict(
            discount=1,
            ensemble_size=3,
            num_bins=20,
            actor_update_interval=1,
            targets_update_interval=5,
            tau=0.05,
            use_distributional_rl=True,
            # Cost configs
            cost_min=0.0,
            cost_max=2.0,
            cost_N=20,
            cost_critic_lr=0.001,
        )

        self.agent = DRLDDPGLag(
            # DDPG args
            state_dim,  # Concatenating obs and goal
            action_dim,
            max_action,
            CriticCls=GoalConditionedCritic,
            **agent_cfg,  # type: ignore
        )
        self.policy = GaussianPolicy(self.agent)

        self.buffer = ConstrainedReplayBuffer(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_size=40,
        )

        self.collector = ConstrainedCollector(
            policy=self.policy,
            buffer=self.buffer,
            env=self.env,
            initial_collect_steps=20,
        )

    def test_step(self):
        self.collector.step(1000)
        self.assertTrue(self.collector.num_eps > 0)

    def test_simple_optimize(self):
        """Test simple train loop without Lagrange"""
        num_iterations = 10
        collect_steps = 2

        num_eps = self.collector.num_eps
        ep_cost = 0
        for i in range(1, num_iterations + 1):
            self.collector.step(collect_steps)
            self.agent.train()
            opt_info = self.agent.optimize(self.buffer, iterations=1, batch_size=64)
            cprint(str(opt_info), "yellow")

            if self.collector.num_eps > num_eps:
                ep_cost = self.collector.past_eps[-1]["ep_cost"]
                ep_len = self.collector.past_eps[-1]["ep_len"]
                cprint(
                    "[INFO] eps Jc='{:.2f}', eps length={}".format(ep_cost, ep_len),
                    "green",
                )
                num_eps = self.collector.num_eps

    def test_eval_agent_n_record_init_states(self):
        self.eval_env = safe_env_load_fn(
            env_kwargs=self.env_kwargs,
            cost_f_kwargs=self.cost_f_kwargs,
            max_episode_steps=10,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            terminate_on_timeout=True,
        )
        num_evals = 5
        _ = ConstrainedCollector.eval_agent_n_record_init_states(
            self.agent, self.eval_env, num_evals
        )


if __name__ == "__main__":
    unittest.main()
