import unittest
from gym.spaces import Box

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.runners.crl_runner import eval_pointenv_cost_constrained_dists
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
)

"""
python pud/algos/unit_tests/test_runner.py TestCRLRunner.test_eval_pointenv_cost_constrained_dists
"""


class TestCRLRunner(unittest.TestCase):
    def setUp(self):
        env_kwargs = {
            "walls": "CentralObstacle",
            "resize_factor": 5,
            "thin": False,
            "cost_limit": 1,
        }
        cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.0,
        }

        self.eval_env = safe_env_load_fn(
            env_kwargs=env_kwargs,
            cost_f_kwargs=cost_f_kwargs,
            max_episode_steps=10,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            terminate_on_timeout=True,
        )

        assert isinstance(self.eval_env.observation_space, dict)
        obs_dim = self.eval_env.observation_space["observation"].shape[0]
        goal_dim = obs_dim
        state_dim = obs_dim + goal_dim
        assert self.eval_env.action_space.shape is not None
        action_dim = self.eval_env.action_space.shape[0]
        assert isinstance(self.eval_env.action_space, Box)
        max_action = float(self.eval_env.action_space.high[0])

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

    def test_eval_pointenv_cost_constrained_dists(self):
        assert isinstance(self.eval_env, SafeGoalConditionedPointWrapper)
        _ = eval_pointenv_cost_constrained_dists(
            agent=self.agent,
            eval_env=self.eval_env,
            sample_args={"sample_key": "ub"},
        )


if __name__ == "__main__":
    unittest.main()
