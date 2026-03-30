import unittest
from termcolor import cprint

from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    set_safe_env_difficulty,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
)

"""
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_type_checking
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_safe_env_load_fn
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_reset_no_constraint
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_reset_with_constraint
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_set_safe_env_difficulty
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_reset_with_constraint_strict_req
"""


class TestSafeWrapper(unittest.TestCase):
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

        self.p_env = SafePointEnv(**env_kwargs, cost_f_args=cost_f_kwargs)

        self.w_env = SafeGoalConditionedPointWrapper(
            self.p_env,
        )

        self.env_kwargs = env_kwargs
        self.cost_f_kwargs = cost_f_kwargs

    def test_type_checking(self):
        blend_env = SafeGoalConditionedPointBlendWrapper(
            env=self.p_env,
            reset_blend=1.0,
            cbfs_policy_path="pud/envs/precompiles/central_obstacle_v2.pkl",
        )
        blend_env.reset()
        assert isinstance(blend_env, SafeGoalConditionedPointBlendWrapper)
        assert isinstance(blend_env, SafeGoalConditionedPointWrapper)
        print("inherited class also belongs to its parent class")

    def test_set_safe_env_difficulty(self):
        set_safe_env_difficulty(self.w_env, 0.5)

    def test_safe_env_load_fn(self):
        """Load env wrappers with args"""
        wrapper_kwargs = [
            dict(cbfs_policy_path="pud/envs/precompiles/central_obstacle_v2.pkl"),
        ]
        out_env = safe_env_load_fn(
            env_kwargs=self.env_kwargs,
            cost_f_kwargs=self.cost_f_kwargs,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            wrapper_kwargs=wrapper_kwargs,
        )

        assert isinstance(out_env, SafeGoalConditionedPointWrapper)
        out_env.set_sample_goal_args(
            prob_constraint=1,
            min_dist=1.0,
            max_dist=10.0,
            min_cost=0.0,
            max_cost=1.0,
        )
        out_env.reset()

    def test_reset_no_constraint(self):
        self.w_env.set_sample_goal_args(
            prob_constraint=0.0,
            min_dist=0.0,
            max_dist=1.0,
            min_cost=0.0,
            max_cost=1.0,
        )
        for _ in range(100):
            _, _ = self.w_env.reset()

    def test_reset_with_constraint(self):
        max_cost = 0.5
        self.w_env.set_sample_goal_args(
            prob_constraint=1.0,
            min_dist=1.0,
            max_dist=10.0,
            min_cost=0.0,
            max_cost=max_cost,
        )
        for _ in range(100):
            out, info = self.w_env.reset()
            self.assertTrue(self.w_env.get_state_cost(out["observation"]) <= max_cost)
            self.assertTrue(self.w_env.get_state_cost(out["goal"]) <= max_cost)

    def test_reset_with_constraint_strict_req(self):
        """Very strict requirement on reference distances, which may be an empty set"""
        target_dists = [2, 5, 10, 15]

        for i in range(len(target_dists)):
            min_dist, max_dist = target_dists[i], target_dists[i]
            cost_levels = list(self.w_env.pi_cbfs["trajs"][min_dist].keys())
            cost_levels.sort()
            for level in range(len(cost_levels)):
                max_cost, min_cost = cost_levels[level], cost_levels[level]

                if max_cost in self.w_env.pi_cbfs["trajs"][min_dist]:
                    self.w_env.set_sample_goal_args(
                        prob_constraint=1.0,
                        min_dist=min_dist,
                        max_dist=max_dist,
                        min_cost=min_cost,
                        max_cost=max_cost,
                    )
                    for _ in range(100):
                        out, info = self.w_env.reset()
                        self.assertTrue(
                            self.w_env.get_state_cost(out["observation"]) <= max_cost
                        )
                        self.assertTrue(
                            self.w_env.get_state_cost(out["goal"]) <= max_cost
                        )

                else:
                    cprint(
                        "[WARN] target cost={}, target distance={} not found".format(
                            cost_levels[level], target_dists[i]
                        )
                    )

    def test_step(self):
        self.w_env.reset()
        at = self.w_env.action_space.sample()
        _, _, _, _ = self.w_env.step(at)

    def test_cbfs_sample(self):
        self.w_env.cbfs_sample(min_cost=0, max_cost=1.0, min_dist=1.0, max_dist=10)


if __name__ == "__main__":
    unittest.main()
