import unittest
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointQueueWrapper,
)
from pud.envs.safe_pointenv.safe_pointenv import (
    SafePointEnv,
    plot_safe_walls,
    plot_maze_grid_points,
)

"""
python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_safe_env_load_fn
python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_plot_safe_walls_w_grids
python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_points_picked_in_obstacle

python -m debugpy \
    --listen localhost:5678 \
    --wait-for-client \
    pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_plot_safe_walls_w_grids
"""


class TestSafePointEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "walls": "Line",
            "resize_factor": 1,
            "thin": False,
        }
        self.cost_f_kwargs = {
            "name": "constant",
            "radius": 1.0,
        }
        self.precompilation_kwargs = {
            "cost_limit": 0,
        }

        self.p_env = SafePointEnv(
            **self.env_kwargs,  # type: ignore
            **self.precompilation_kwargs,
            cost_f_args=self.cost_f_kwargs
        )

    def test_safe_env_load_fn(self):
        """Test env loader with wrappers
        TimeLimit is loaded by default if max_episode_steps>0
        """
        env_args = deepcopy(self.env_kwargs)
        env_args.update(self.precompilation_kwargs)
        gym_env_wrappers = [SafeGoalConditionedPointQueueWrapper]
        env = safe_env_load_fn(
            env_args,
            self.cost_f_kwargs,
            max_episode_steps=20,
            gym_env_wrappers=gym_env_wrappers,
            terminate_on_timeout=False,
        )

        result = env.reset()
        if isinstance(result, tuple):
            _, info = result
        else:
            info = {}
        self.assertTrue("cost" in info)

        num_steps = int(1e6)
        for _ in tqdm(range(int(1e6)), total=num_steps):
            at = env.action_space.sample()
            _, _, _, info = env.step(at)
            self.assertTrue("cost" in info)

    def test_plot_safe_walls(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            ax = plot_safe_walls(
                walls=self.p_env._walls,
                cost_map=self.p_env._cost_map,
                cost_limit=cost_ub,
                ax=ax,
            )
            fig.savefig(
                output_dir.joinpath(
                    "{}_resize={:0>2d}_cost={:.2f}.jpg".format(
                        self.p_env.wall_name, self.p_env.resize_factor, cost_ub
                    )
                ),
                dpi=300,
            )
            plt.close(fig)

    def test_plot_safe_walls_w_grids(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        # For cost_ub in [0, 1, 2]:
        for cost_ub in [0]:
            fig, ax = plt.subplots()
            ax = plot_maze_grid_points(walls=self.p_env._walls, ax=ax)
            ax = plot_safe_walls(
                walls=self.p_env._walls,
                cost_map=self.p_env._cost_map,
                cost_limit=cost_ub,
                ax=ax,
            )
            fig.savefig(
                output_dir.joinpath(
                    "{}_resize={:0>2d}_cost={:.2f}_w_grids.jpg".format(
                        self.p_env.wall_name, self.p_env.resize_factor, cost_ub
                    )
                ),
                dpi=300,
            )
            plt.close(fig)

    def test_points_picked_in_obstacle(self):
        """All points are picked on top of obstacle, they should all be blocked"""
        self.env_kwargs["walls"] = "LQuarter"
        self.p_env = SafePointEnv(
            **self.env_kwargs,  # type: ignore
            **self.precompilation_kwargs,
            cost_f_args=self.cost_f_kwargs
        )
        pnts = np.loadtxt(
            "pud/envs/safe_pointenv/unit_tests/LQuarter_resize_5_blocks.txt",
            delimiter=",",
        )
        pnts_orig = pnts * np.array([self.p_env._height, self.p_env._width])
        for p in pnts_orig:
            assert self.p_env._is_blocked(p)

    def test_reset(self):
        for _ in range(100):
            reset_result = self.p_env.reset()
            if reset_result is None:
                continue
            new_state, _ = reset_result
            sample_cost = self.p_env.get_state_cost(new_state)
            cx, cy = new_state
            self.assertTrue(
                sample_cost < self.p_env.cost_limit,
                msg="sample={}, sample cost= {}, cost map = {}, cost limit={}".format(
                    new_state,
                    sample_cost,
                    self.p_env._cost_map[int(cx), int(cy)],
                    self.p_env.cost_limit,
                ),
            )
            self.assertTrue(not self.p_env._is_blocked(new_state))

    def test_step(self):
        reset_result = self.p_env.reset()
        if reset_result is None:
            return
        else:
            _, _ = reset_result
        at = self.p_env.action_space.sample()
        self.p_env.step(at)


if __name__ == "__main__":
    unittest.main()
