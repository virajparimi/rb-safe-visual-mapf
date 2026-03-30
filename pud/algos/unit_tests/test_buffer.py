import unittest

from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper

"""
python pud/algos/unit_tests/test_buffer.py TestConstrainedBuffer.test_add
"""


class TestConstrainedBuffer(unittest.TestCase):
    def setUp(self):
        env_kwargs = {
            "walls": "CentralObstacle",
            "resize_factor": 5,
            "thin": False,
        }
        cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.0,
        }
        precompilation_kwargs = {
            "cost_limit": 1,
        }

        self.p_env = SafePointEnv(
            **env_kwargs, **precompilation_kwargs, cost_f_args=cost_f_kwargs  # type: ignore
        )

        self.w_env = SafeGoalConditionedPointWrapper(self.p_env)

    def test_add(self):
        out, info = self.w_env.reset()
        obs_dim = len(out["observation"])
        goal_dim = obs_dim
        assert self.w_env.action_space.shape is not None
        action_dim = self.w_env.action_space.shape[0]
        buffer = ConstrainedReplayBuffer(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_size=10,
        )
        for _ in range(50):
            state, info = self.w_env.reset()
            at = self.w_env.action_space.sample()
            next_state, rew, done, info = self.w_env.step(at)
            buffer.add(
                state=state,
                action=at,
                next_state=next_state,
                reward=rew,
                cost=float(info["cost"]),
                done=bool(done),
            )
        self.assertTrue(buffer.size == 10)


if __name__ == "__main__":
    unittest.main()
