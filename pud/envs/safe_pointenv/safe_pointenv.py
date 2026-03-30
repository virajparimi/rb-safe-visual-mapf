import time
import numpy as np
from tqdm.auto import tqdm
from gym.spaces import Box
from numpy.typing import NDArray
from matplotlib.axes import Axes
from typing import Optional, Union

from pud.envs.simple_navigation_env import PointEnv


def plot_safe_walls(
    walls: np.ndarray,
    cost_map: Optional[np.ndarray],
    cost_limit: Optional[float],
    ax: Axes,
) -> Axes:
    """
    Visualize the map in range from 0 to N+1 (+1 from the array dimension)
    """
    (height, width) = walls.shape
    # only plot walls
    for i, j in zip(*np.where(walls)):
        x = np.array([i, i + 1]) / float(height)
        y0 = np.array([j, j]) / float(width)
        y1 = np.array([j + 1, j + 1]) / float(width)
        ax.fill_between(x, y0, y1, color="grey")

    # Scattered points are more accurate as they are state-wise estimations
    if cost_map is not None:
        unsafe_points = np.where(cost_map > cost_limit)
        unsafe_points = np.column_stack(unsafe_points)
        ax.scatter(
            unsafe_points[:, 0] / float(height),
            unsafe_points[:, 1] / float(width),
            s=2,
            marker="o",
            c="red",
        )

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_start_and_goals(
    walls: np.ndarray,
    ax: Axes,
    starts: NDArray = np.array([]),
    goals: NDArray = np.array([]),
    s: int = 40,
    start_color="g",
    goal_color="r",
    zorder=5,
    normalize=True,
) -> Axes:
    (height, width) = walls.shape
    starts = np.array(starts)
    if normalize:
        starts = starts / np.array([[height, width]])
    ax.scatter(
        starts[:, 0],
        starts[:, 1],
        color=start_color,
        zorder=zorder,
        marker="s",
        label="start",
        s=s,
    )

    goals = np.array(goals)
    if normalize:
        goals = goals / np.array([[height, width]])
    ax.scatter(
        goals[:, 0],
        goals[:, 1],
        color=goal_color,
        zorder=zorder,
        marker="x",
        label="goal",
        s=s,
    )

    return ax


def plot_trajs(
    list_trajs,
    walls: np.ndarray,
    ax: Axes,
    starts: NDArray = np.array([]),
    goals: NDArray = np.array([]),
    s: int = 40,
    start_color: str = "#18aedb",
    goal_color: str = "#dbbb18",
    traj_color: Optional[str] = None,
    use_pbar=False,
) -> Axes:
    pbar = tqdm(total=len(list_trajs), disable=(not use_pbar))

    (height, width) = walls.shape

    """Plot a list of trajs, each is a list of tuples (int states)"""
    traj_starts = []
    traj_goals = []

    for traj in list_trajs:
        # Randomize colors
        c = traj_color
        if traj_color is None:
            c = np.random.rand(
                3,
            )
        for i in range(0, len(traj) - 1):
            pnt = traj[i]
            pnt_next = traj[i + 1]
            x, y = pnt[0] / float(height), pnt[1] / float(width)
            xn, yn = pnt_next[0] / float(height), pnt_next[1] / float(width)
            ax.plot([x, xn], [y, yn], marker="o", color=c, markersize=4, alpha=0.5)

            if i == 0:
                traj_starts.append([x, y])
            if i == len(traj) - 2:
                traj_goals.append([xn, yn])

        pbar.update()

    if len(starts) == 0:
        starts = np.array(traj_starts)
    else:
        # the externally supplied starts need to be flipped
        starts = np.array(starts) / np.array([[height, width]])

    ax.scatter(
        starts[:, 0],
        starts[:, 1],
        color=start_color,
        zorder=5,
        marker="s",
        label="start",
        s=s,
    )

    if len(goals) == 0:
        goals = np.array(traj_goals)
    else:
        goals = np.array(goals) / np.array([[height, width]])

    ax.scatter(
        goals[:, 0],
        goals[:, 1],
        color=goal_color,
        zorder=5,
        marker="x",
        label="goal",
        s=s,
    )

    pbar.close()
    return ax


def plot_maze_grid_points(walls: np.ndarray, ax: Axes) -> Axes:
    (height, width) = walls.shape
    empty_points = np.where(walls == 0)
    empty_points = np.column_stack(empty_points)
    ax.scatter(
        empty_points[:, 0] / float(height),
        empty_points[:, 1] / float(width),
        s=0.5,
        marker="o",
        c="green",
    )
    return ax


class SafePointEnv(PointEnv):
    """
    - Ensure start states are always safe.
    - In each step, a cost is returned along with other info
    - Rapidly estimate upper and lower feasible trajectory cost

    NOTE: To allow rapid estimation of trajectory cost, make sure there is zero-cost trajectory between each test
    start and goal states. Use plot_safe_walls method to visualize the zero step cost map.
    """

    def __init__(
        self,
        walls: Union[str, None] = None,
        resize_factor: int = 1,
        action_noise=1.0,
        thin=False,
        # Cost configs
        cost_f_args: dict = {},
        cost_limit: float = 0.5,
        verbose: bool = True,
    ):
        t0 = time.time()
        super(SafePointEnv, self).__init__(
            walls,
            resize_factor,
            action_noise,
            thin,
        )
        if verbose:
            print("[INFO] PointEnv setup: {} s".format(time.time() - t0))

        self.resize_factor = resize_factor
        self.thin = thin
        self.wall_name = walls
        self.cost_limit = cost_limit

        obstacle_x, obstacle_y = np.where(self._walls == 1)
        self.obstacles = np.stack([obstacle_x, obstacle_y], axis=-1).astype(
            float
        )  # N x 2

        self.cost_f_cfg = cost_f_args
        cost_fn_name = cost_f_args.get("name")
        self.cost_function = None

        t0 = time.time()

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

        self.safe_empty_states = self.gather_safe_empty_states(self.cost_limit)
        self.reset()
        print("[INFO] SafePointEnv setup: {} s".format(time.time() - t0))

    def get_map(self):
        return self._walls

    def get_cost_map(self):
        return self._cost_map

    def get_map_width(self):
        return self._width

    def get_map_height(self):
        return self._height

    def get_internal_state(self):
        return self.state.copy()

    def set_cost_limit(self, cost_limit: float):
        self.cost_limit = cost_limit

    def build_cost_map(self):
        (height, width) = self._walls.shape
        cost_map = np.ones([height + 1, width + 1], dtype=float) * np.inf

        assert self.cost_function is not None, "Cost function is not set"
        for i in range(cost_map.shape[0]):
            for j in range(cost_map.shape[1]):
                min_d, _ = self.dist_2_blocks(np.array([i, j]))
                cost_map[i, j] = self.cost_function(min_d)
        return cost_map

    def reset(self):
        if (not hasattr(self, "cost_limit")) or (not hasattr(self, "_cost_map")):
            print(
                "[INFO] skipping the reset in PointEnv.__init__ because setup is not ready yet"
            )
            return

        # TODO: Perhaps suffer from label inbalance?
        self.state = self.sample_safe_empty_state()
        new_state_cost = self.get_state_cost(xy=self.state)
        info = {"cost": new_state_cost}
        return self.state.copy(), info

    def reset_manual(self, start_state: np.ndarray):
        "Manually set the start state"
        self.state = start_state
        new_state_cost = self.get_state_cost(xy=self.state)
        info = {"cost": new_state_cost}
        return self.state.copy(), info

    def gather_safe_empty_states(self, cost_limit: float):
        """
        Due to the increased cost in reset, precompile a list of initial states here
        """
        empty_states = np.where(self._walls == 0)
        safe_empty_states = [[], []]

        for cx, cy in zip(*empty_states):
            # Only sample states whose costs are lower than an upper bound
            if self._cost_map[cx, cy] <= cost_limit:
                safe_empty_states[0].append(cx)
                safe_empty_states[1].append(cy)

        safe_empty_states = np.column_stack(safe_empty_states)  # N,d
        return safe_empty_states

    def sample_safe_empty_state(self):
        """
        Must take intersection with the empty states because state cost is computed from the center of the block?
        """
        num_candidate_states = len(self.safe_empty_states)

        idx = np.random.randint(0, num_candidate_states)
        new_state = self.safe_empty_states[idx].astype(np.float32)

        # Don't remove the checks below
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

    def get_state_cost(self, xy: NDArray):
        assert self.cost_function is not None, "Cost function is not set"
        min_d, _ = self.dist_2_blocks(xy)
        return self.cost_function(min_d)

    def step(self, action):
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        assert isinstance(self.action_space, Box)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        # NOTE: Use the maximum cost along the action segment
        cost = self.get_state_cost(self.state)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state
                    new_cost = self.get_state_cost(new_state)
                    if cost < new_cost:
                        cost = new_cost

        assert not self._is_blocked(
            self.state
        ), "New state is in collision, might be a bug"
        done = False
        rew = -1.0
        return self.state.copy(), rew, done, {"cost": cost}
