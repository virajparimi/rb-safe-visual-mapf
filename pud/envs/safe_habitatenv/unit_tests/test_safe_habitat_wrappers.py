import unittest

from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
    SafeGoalConditionedHabitatPointWrapper,
)

"""
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitat_wrappers.py TestSafeHabitatWrapper.safe_habitat_env_load_fn
"""


class TestSafeHabitatWrapper(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = dict(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
        )
        self.cost_kwargs = dict(
            cost_f_args={"name": "linear", "radius": 2.0},
            cost_limit=10.0,
        )

    def safe_habitat_env_load_fn(self):
        env = safe_habitat_env_load_fn(
            env_kwargs=self.env_kwargs,
            **self.cost_kwargs,  # type: ignore
            gym_env_wrappers=(SafeGoalConditionedHabitatPointWrapper),  # type: ignore
        )
        reset_result = env.reset()
        if reset_result is not None:
            _, _ = reset_result
        for _ in range(100):
            action = env.action_space.sample()
            _, _, _, _ = env.step(action)


if __name__ == "__main__":
    unittest.main()
