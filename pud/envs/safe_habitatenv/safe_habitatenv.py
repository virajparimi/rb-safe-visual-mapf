import time
import argparse
import numpy as np
from gym.spaces import Box
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Dict, Union, Literal

from pud.visualizers.visualize import plot_safe_walls
from pud.envs.habitat_navigation_env import HabitatNavigationEnv


class SafeHabitatNavigationEnv(HabitatNavigationEnv):
    def __init__(
        self,
        env_type: Literal["HabitatSim", "ReplicaCAD"],
        height: Union[float, None] = None,
        action_noise: float = 1.0,
        simulator_settings: dict = {},
        sensor_type: Literal["rgb", "depth"] = "depth",
        apsp_path: str = "",
        device: str = "cpu",
        # Cost specific arguments
        cost_f_args: dict = {},
        cost_limit: float = 0.5,
    ):

        super().__init__(
            env_type=env_type,
            height=height,
            action_noise=action_noise,
            simulator_settings=simulator_settings,
            sensor_type=sensor_type,
            apsp_path=apsp_path,
            device=device,
        )

        self.cost_function = None
        self.cost_limit = cost_limit
        self.cost_f_cfg = cost_f_args

        obstacles_x, obstacles_y = np.where(self._walls == 1)
        self.obstacles = np.stack([obstacles_x, obstacles_y], axis=-1).astype(float)

        t0 = time.time()

        cost_fn_name = self.cost_f_cfg.get("name")
        self.cost_function = None
        if cost_fn_name:
            import functools
            from pud.envs.safe_pointenv.cost_functions import (
                cost_from_cosine_distance,
                cost_from_linear_distance,
                const_cost_from_distance,
            )

            if cost_fn_name == "cosine":
                self.cost_function = functools.partial(
                    cost_from_cosine_distance, r=self.cost_f_cfg["radius"]
                )
            elif cost_fn_name == "linear":
                self.cost_function = functools.partial(
                    cost_from_linear_distance, r=self.cost_f_cfg["radius"]
                )
            elif cost_fn_name == "constant":
                self.cost_function = functools.partial(
                    const_cost_from_distance, r=self.cost_f_cfg["radius"]
                )
            else:
                raise Exception("Unsupported cost function")

            # NOTE: cost map is computed based on states, not trajectories/accumulated costs
            self._cost_map = self.build_cost_map()

        self._safe_empty_states = self.gather_safe_empty_states()
        self.reset()
        print("[INFO] SafeHabitatNavigationEnv Setup: {} s".format(time.time() - t0))

    def get_map(self):
        return self._walls

    def get_cost_map(self):
        return self._cost_map

    def get_map_width(self):
        return self._wall_width

    def get_map_height(self):
        return self._wall_height

    def get_internal_state(self):
        assert self.state_grid is not None
        return self.state_grid.copy()

    def set_cost_limit(self, cost_limit: float):
        self.cost_limit = cost_limit

    @property
    def cost_map(self):
        return self._cost_map

    def build_cost_map(self):
        (height, width) = self._walls.shape
        cost_map = np.ones([height + 1, width + 1], dtype=float) * np.inf

        assert self.cost_function is not None, "Cost function is not set"
        for i in range(cost_map.shape[0]):
            for j in range(cost_map.shape[1]):
                min_d, _ = self.dist_2_blocks(np.array([i, j]))
                cost_map[i, j] = self.cost_function(min_d)
        return cost_map

    def gather_safe_empty_states(self) -> NDArray:
        """
        Due to the increased cost in reset, precompile a list of initial states here
        """
        empty_states = np.where(self._walls == 0)
        safe_empty_states = [[], []]

        for cx, cy in zip(*empty_states):
            # Only sample states whose costs are lower than an upper bound
            if self._cost_map[cx, cy] <= self.cost_limit:
                safe_empty_states[0].append(cx)
                safe_empty_states[1].append(cy)

        safe_empty_states = np.column_stack(safe_empty_states).astype(np.float32)  # N x d

        return safe_empty_states

    def sample_safe_empty_state(self):
        """
        Must take the intersection with the empty states because state cost is computed from the center of the block
        """
        num_candidate_states = len(self._safe_empty_states)

        idx = np.random.randint(0, num_candidate_states)
        new_state = self._safe_empty_states[idx]

        # NOTE: Don't remove the checks below
        assert not self._is_blocked(new_state)
        assert self.get_state_cost(new_state) <= self.cost_limit
        return new_state

    def dist_2_blocks(self, xy: NDArray):
        """
        Calculate the distance between a float state xy and a block state that are ints (from array indices)

        a block covers an square area of
        block_x -- block_x+1
        block_y -- block_y+1

        Args:
            xy (np.ndarray): [x,y]
            block_xys (np.ndarray): [[block_x, block_y], ... ]

        Returns:
            float: calculated distance
            int: index of the nearest block

        Example:
            xy = np.array([0.5, 0.6])

            block_xys = np.array([[0,1],[2,5]])

        Reference:
        https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """

        bxs_min = self.obstacles[:, 0]
        bys_min = self.obstacles[:, 1]
        x, y = xy

        dxs = np.maximum(bxs_min - x, x - (bxs_min + 1))
        dxs = np.maximum(dxs, 0.0)

        dys = np.maximum(bys_min - y, y - (bys_min + 1))
        dys = np.maximum(dys, 0)

        d2 = dxs**2.0 + dys**2.0
        ind_min = np.argmin(d2)
        d_min = np.sqrt(d2[ind_min])

        return d_min, ind_min

    def get_state_cost(self, xy: NDArray) -> float:
        """
        Assumes that the xy argument is the grid position and not the continuous position
        """
        assert self.cost_function is not None, "Cost function is not set"
        min_d, _ = self.dist_2_blocks(xy)
        return float(self.cost_function(min_d))

    def seed(self, seed: int) -> None:
        self._simulator.seed(seed)

    def reset(self):
        if (not hasattr(self, "cost_limit")) or (not hasattr(self, "_cost_map")):
            print(
                "[INFO] Skipping the reset in HabitatNavigationEnv.__init__ because setup is not ready yet"
            )
            return
        self.state_grid = self.sample_empty_state()
        assert self.state_grid is not None
        agent_cost = self.get_state_cost(xy=np.array(self.state_grid))
        info = {"cost": agent_cost}
        return self.state_grid.copy(), info

    def reset_manual(self, start_state: np.ndarray):
        "manually set the start state"
        self.state_grid = start_state
        new_state_cost = self.get_state_cost(xy=self.state_grid)
        info = {"cost": new_state_cost}
        return self.state_grid.copy(), info

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, Dict]:
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        assert isinstance(self.action_space, Box)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        # NOTE: Use the maximum cost along the action segment
        assert self.state_grid is not None
        start_state = self.state_grid.copy()
        cost = self.get_state_cost(start_state)

        assert not self._is_blocked(start_state)

        num_substeps = 10
        dt = 1.0 / num_substeps

        # Same as the point env, check each axis individually to allow more movement
        num_axis = len(action)
        dt = 1.0 / num_substeps
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state_grid.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state_grid = new_state
                    new_cost = self.get_state_cost(new_state)
                    cost = max(new_cost, cost)

        done = False
        rew = -1.0
        return self.state_grid.copy(), rew, done, {"cost": cost}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = SafeHabitatNavigationEnv(
        env_type="ReplicaCAD",
        sensor_type="rgb",
        device="cuda:1",
        cost_f_args={"name": "cosine", "radius": 2.0},
        cost_limit=0.0,
    )

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax = plot_safe_walls(env.walls, "map_visual", env.cost_map, env.cost_limit, ax=ax)  # type: ignore
    plt.savefig("temp/visual.jpg", dpi=300)
