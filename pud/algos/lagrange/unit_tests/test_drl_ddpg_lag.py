import torch
import unittest

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer

"""
python pud/algos/lagrange/unit_tests/test_drl_ddpg_lag.py TestDRLDDPGLag.test_lagrange
"""


class TestDRLDDPGLag(unittest.TestCase):
    def setUp(self):
        obs_dim, goal_dim = 2, 2
        action_dim = 2
        max_action = 1.0
        state_dim = obs_dim + goal_dim

        self.env_kwargs = {
            "walls": "CentralObstacle",
            "resize_factor": 5,
            "thin": False,
            "cost_limit": 1,
        }
        cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.0,
        }

        self.p_env = SafePointEnv(**self.env_kwargs, cost_f_args=cost_f_kwargs)

        self.w_env = SafeGoalConditionedPointWrapper(self.p_env)

        self.agent = DRLDDPGLag(
            # DDPG args
            state_dim,
            action_dim,
            max_action,
            discount=1,
            actor_update_interval=1,
            targets_update_interval=1,
            tau=0.005,
            CriticCls=GoalConditionedCritic,
            # UVFDDPG args
            num_bins=20,
            use_distributional_rl=True,
            ensemble_size=3,
            # Cost configs
            cost_min=0,
            cost_max=2.0,
            cost_N=20,
            cost_critic_lr=1e-3,
            cost_limit=self.env_kwargs["cost_limit"],
            lambda_lr=0.001,
            lambda_optimizer="Adam",
        )

        out, info = self.w_env.reset()
        obs_dim = len(out["observation"])
        goal_dim = obs_dim
        assert self.w_env.action_space.shape is not None
        action_dim = self.w_env.action_space.shape[0]
        self.buffer = ConstrainedReplayBuffer(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_size=10,
        )

        for _ in range(250):
            state, info = self.w_env.reset()
            at = self.w_env.action_space.sample()
            next_state, rew, done, info = self.w_env.step(at)
            self.buffer.add(
                state=state,
                action=at,
                next_state=next_state,
                reward=rew,
                cost=float(info["cost"]),
                done=bool(done),
            )

    def test_lagrange(self):
        self.assertTrue(self.agent.lagrange.cost_limit == self.env_kwargs["cost_limit"])
        self.assertTrue(
            isinstance(self.agent.lagrange.lambda_optimizer, torch.optim.Adam)
        )

    def test_cost_critic_loss(self):
        """only test if the cost loss runs OK, but not the math"""
        for _ in range(10):

            # Each of these are batches
            state, next_state, action, _, cost, done = self.buffer.sample_w_cost(20)

            current_q = self.agent.cost_critic(state, action)
            target_q = self.agent.cost_critic_target(
                next_state, self.agent.actor_target(next_state)
            )
            _ = self.agent.cost_critic_loss(
                current_q=current_q,
                target_q=target_q,
                cost=cost,
                done=done,
            )

    def test_get_cost_q_values(self):
        state, _, _, _, _, _ = self.buffer.sample_w_cost(20)
        self.agent.get_cost_q_values(state=state, aggregate="mean")

    def test_get_cost_to_goal(self):
        state, _, _, _, _, _ = self.buffer.sample_w_cost(20)
        self.agent.get_cost_to_goal(state=state, aggregate="mean")

    def test_optimize(self):
        self.agent.optimize(replay_buffer=self.buffer, iterations=1, batch_size=20)

    def test_optimize_lagrange(self):
        self.agent.optimize_lagrange(ep_cost=2.0)


if __name__ == "__main__":
    unittest.main()
