import gym
import time
import yaml
import pickle
import random
import habitat_sim
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Literal
from tqdm.auto import tqdm
from termcolor import colored
from matplotlib.axes import Axes
from gym.spaces import Box, Dict
from numpy.typing import NDArray
from typing import Tuple, Union, List, Optional

# Define data types
TypeGridXY = NDArray
TypeHabitatXY = NDArray
TypeHabitatXYZ = NDArray
TypeHabitatSensorObs = NDArray


def make_habitat_configuration(
    settings: dict,
    sensor_type: Literal["rgb", "depth"] = "depth",
    device: str = "cpu",
):
    simulator_cfg = habitat_sim.SimulatorConfiguration()

    simulator_cfg.scene_id = settings["scene"]

    if "scene_dataset" in settings:
        simulator_cfg.scene_dataset_config_file = settings["scene_dataset"]

    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        simulator_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        simulator_cfg.override_scene_light_defaults = settings[
            "override_scene_light_defaults"
        ]
    if "scene_light_setup" in settings:
        simulator_cfg.scene_light_setup = settings["scene_light_setup"]

    if device == "cpu":
        simulator_cfg.gpu_device_id = -1

    # Generate a default RBG camera specification and attach it to each robot
    rgb_sensor_spec_forward = habitat_sim.sensor.CameraSensorSpec()
    rgb_sensor_spec_forward.uuid = "color_sensor_forward"
    if sensor_type == "rgb":
        rgb_sensor_spec_forward.sensor_type = habitat_sim.sensor.SensorType.COLOR
    elif sensor_type == "depth":
        rgb_sensor_spec_forward.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    rgb_sensor_spec_forward.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec_forward.position = [
        0.0,
        settings["sensor_height"],
        0.0,
    ]

    rgb_sensor_spec_right = habitat_sim.sensor.CameraSensorSpec()
    rgb_sensor_spec_right.uuid = "color_sensor_right"
    if sensor_type == "rgb":
        rgb_sensor_spec_right.sensor_type = habitat_sim.sensor.SensorType.COLOR
    elif sensor_type == "depth":
        rgb_sensor_spec_right.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    rgb_sensor_spec_right.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec_right.position = [
        0.0,
        settings["sensor_height"],
        0.0,
    ]
    rgb_sensor_spec_right.orientation = [0.0, -np.pi / 2, 0.0]

    rgb_sensor_spec_backward = habitat_sim.sensor.CameraSensorSpec()
    rgb_sensor_spec_backward.uuid = "color_sensor_backward"
    if sensor_type == "rgb":
        rgb_sensor_spec_backward.sensor_type = habitat_sim.sensor.SensorType.COLOR
    elif sensor_type == "depth":
        rgb_sensor_spec_backward.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    rgb_sensor_spec_backward.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec_backward.position = [
        0.0,
        settings["sensor_height"],
        0.0,
    ]
    rgb_sensor_spec_backward.orientation = [0.0, np.pi, 0.0]

    rgb_sensor_spec_left = habitat_sim.sensor.CameraSensorSpec()
    rgb_sensor_spec_left.uuid = "color_sensor_left"
    if sensor_type == "rgb":
        rgb_sensor_spec_left.sensor_type = habitat_sim.sensor.SensorType.COLOR
    elif sensor_type == "depth":
        rgb_sensor_spec_left.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    rgb_sensor_spec_left.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec_left.position = [
        0.0,
        settings["sensor_height"],
        0.0,
    ]
    rgb_sensor_spec_left.orientation = [0.0, np.pi / 2, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        rgb_sensor_spec_forward,
        rgb_sensor_spec_right,
        rgb_sensor_spec_backward,
        rgb_sensor_spec_left,
    ]

    return habitat_sim.Configuration(simulator_cfg, [agent_cfg])


def get_default_habitat_sim_settings(env_type: Literal["HabitatSim", "ReplicaCAD"]):
    with open("configs/habitat_data.yaml", "r") as f:
        habitat_data_f = yaml.safe_load(f)
        return habitat_data_f[env_type]["default_settings"]


class HabitatNavigationEnv(gym.Env):
    def __init__(
        self,
        env_type: Literal["HabitatSim", "ReplicaCAD"],
        scene: Optional[str] = None,
        height: Optional[float] = None,
        action_noise: float = 1.0,
        simulator_settings: dict = {},
        sensor_type: Literal["rgb", "depth"] = "depth",
        apsp_path: str = "",
        device: str = "cpu",
    ):
        """
        initialize the habitat env and extract the 2D grid map
        """

        self.sensor_type = sensor_type
        self.device = device

        if not simulator_settings:
            print(
                "[{}]: using default setting for {}".format(
                    colored("INFO", "green"), colored(env_type, "red")
                )
            )
            self._simulator_settings = get_default_habitat_sim_settings(
                env_type=env_type
            )
        else:
            self._simulator_settings = simulator_settings
            assert "scene" in self._simulator_settings
            assert "width" in self._simulator_settings
            assert "height" in self._simulator_settings
            assert "default_agent" in self._simulator_settings
            assert "sensor_height" in self._simulator_settings

        if scene:
            # Override the scene
            self._simulator_settings["scene"] = scene

        self._action_noise = action_noise

        # Height and Width of the camera resolution!
        self._width = self._simulator_settings["width"]
        self._height = self._simulator_settings["height"]

        self._configuration = make_habitat_configuration(
            self._simulator_settings,
            sensor_type=sensor_type,
            device=device,
        )

        self._simulator = habitat_sim.Simulator(self._configuration)

        self._agent = self._simulator.initialize_agent(
            self._simulator_settings["default_agent"]
        )

        self.observation_space = Box(
            low=0, high=255, shape=(4, self._height, self._width, 4), dtype=np.uint8  # type: ignore
        )

        # The channels are RGBA
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # Ensure that the pathfinder utility of the simulator is loaded
        assert self._simulator.pathfinder.is_loaded

        # Simulator's top-down visualizer's parameters
        self._meters_per_pixel = 0.4
        if height is not None:
            self._vertical_slice = height
        else:
            self._vertical_slice = self._simulator.pathfinder.get_bounds()[0][1]
        self._walls = (
            self._simulator.pathfinder.get_topdown_view(
                self._meters_per_pixel, self._vertical_slice
            )
            .astype(np.uint8)
            .copy()
        )
        self._walls = 1 - self._walls
        # The original map before uint8 is True/False binary map
        # After conversion, it is 1/0 map

        self._wall_height, self._wall_width = self._walls.shape

        self.grid_observation_space = Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self._wall_height, self._wall_width]),
            dtype=np.float32,
        )

        # Discrete maze all-pair-shortest-path load or calculation
        t0 = time.time()
        path_apsp = None
        if len(apsp_path) > 0:
            path_apsp = Path(apsp_path)
        if path_apsp and Path(apsp_path).exists():
            print("[INFO] loading prior apsp pickle: {}".format(path_apsp.as_posix()))
            with open(apsp_path, "rb") as f:
                self._apsp = pickle.load(f)
        else:
            print("[INFO] Calling the APSP construction function")
            self._apsp = self.compute_apsp(self._walls)
            print("APSP construction time in (s): ", time.time() - t0)
            if path_apsp and (not path_apsp.exists()):
                path_apsp.parent.mkdir(exist_ok=True, parents=True)
                path_apsp_dump = path_apsp.as_posix()
                print("[INFO] saving apsp to {}".format(path_apsp_dump))
                with open(path_apsp_dump, "wb") as f:
                    pickle.dump(self._apsp, f)

        self.reset()

    @property
    def walls(self):
        """1 is obstacle, 0 is empty,
        init flips the values"""
        return self._walls

    @property
    def wall_height(self):
        return self._wall_height

    @property
    def wall_width(self):
        return self._wall_width

    # ----------- Point Env equivalent internal functions ------------------------
    def compute_apsp(self, walls: NDArray):
        """
        NOTE: walls[i, j] is True if (i, j) is traversable and False otherwise
        """
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.inf)
        for (i1, j1), dist_dict in tqdm(
            nx.shortest_path_length(g), total=len(g.nodes())
        ):
            for (i2, j2), d in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist

    def _discretize_state(self, state: TypeGridXY):
        (i, j) = np.floor(state).astype(np.int32)
        # Round down to the nearest cell if at the boundary.
        if i == self._wall_height:
            i -= 1
        if j == self._wall_width:
            j -= 1
        return (i, j)

    def _is_blocked(self, state: TypeGridXY) -> bool:
        """
        check occupancy through 2D maze matrix, the same as the point env
        """
        assert self.grid_observation_space.dtype == state.dtype, "mismatch data type"
        if not self.grid_observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return self._walls[i, j] == 1

    def sample_empty_state(self, max_attempts=100):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        for _ in range(max_attempts):
            state_index = np.random.choice(num_candidate_states)
            state_grid = np.array(
                [candidate_states[0][state_index], candidate_states[1][state_index]],
                dtype=np.float32,
            )
            state_grid += np.random.uniform(size=2)
            if not self._is_blocked(state_grid):
                return state_grid

    def reset(self) -> TypeGridXY:
        return self.reset_in_grid()

    def reset_in_grid(self):
        """reset in grid maze"""
        self.state_grid = self.sample_empty_state()
        assert self.state_grid is not None
        return self.state_grid.copy()

    def get_distance(self, obs: TypeGridXY, goal: TypeGridXY):
        """Compute the shortest path distance.

        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return self._apsp[i1, j1, i2, j2]

    def step(self, action: NDArray):
        return self.step_in_grid(action=action)

    def step_in_grid(self, action: NDArray):
        """
        a step function in 2d grid similar to simple navigation env
        """
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise, 2).astype(action.dtype)
        assert isinstance(self.action_space, Box)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        num_substeps = 10
        assert self.state_grid is not None
        start_state = self.state_grid.copy()
        assert not self._is_blocked(start_state)

        # Same as the point env, check each axis individually to allow more movement
        num_axis = len(action)
        dt = 1.0 / num_substeps
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state_grid.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state_grid = new_state

        done = False
        rew = -1.0
        return self.state_grid.copy(), rew, done, {}

    def get_sensor_obs_at_grid_xy(self, state_grid: TypeGridXY):
        habitat_xy = self.get_habitat_xy_from_grid_xy(state_grid)
        self.set_agent_pos_w_habitatxy(np.array(habitat_xy))
        return self.get_sensor_observation()

    # ---- Interface between Habitat Env --------

    def seed(self, seed: int) -> None:
        self._simulator.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_distance_from_habitat_xy(self, position: NDArray, goal: NDArray) -> float:
        """Compute the shortest path distance.

        NOTE: This distance is *not* used for training. Further, the position and the goal arguments represent
        the 2D coordinates ([x, y])

        """
        (i1, j1) = self.get_grid_xy_from_habitat_xy(position)
        (i2, j2) = self.get_grid_xy_from_habitat_xy(goal)
        return self._apsp[i1, j1, i2, j2]

    def get_grid_xy_from_habitat_xy(self, position: NDArray) -> Tuple[int, int]:
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)

        NOTE: position argument represents the 2D coordinates ([x, y])
        """
        lower_bound, upper_bound = self._simulator.pathfinder.get_bounds()

        grid_x = (
            ((position[0] - lower_bound[2]) / self._meters_per_pixel)
            .round()
            .astype(int)
        )
        grid_y = (
            ((position[1] - lower_bound[0]) / self._meters_per_pixel)
            .round()
            .astype(int)
        )
        return grid_x, grid_y

    def get_habitat_xy_from_grid_xy(
        self, grid_position: Union[Tuple[int, int], Tuple[float, float], NDArray]
    ) -> Tuple[float, float]:

        lower_bound, upper_bound = self._simulator.pathfinder.get_bounds()

        realworld_x = lower_bound[2] + grid_position[0] * self._meters_per_pixel
        realworld_y = lower_bound[0] + grid_position[1] * self._meters_per_pixel
        return realworld_x, realworld_y

    def convert_xyz_to_xy_in_habitat(self, position: NDArray) -> NDArray:
        """
        Convert the simulation 3D coordinates ([y, z, x]) to grid coordinates ([x, y])
        """
        return np.array([position[2], position[0]])

    def convert_xy_to_xyz_in_habitat(self, position: NDArray) -> NDArray:
        """
        Convert the grid coordinates ([x, y]) to simulation 3D coordinates ([y, z, x])
        """
        return np.array([position[1], self._vertical_slice, position[0]])

    def is_blocked_habitat(self, obs: NDArray) -> bool:
        """
        check occupancy through habitat
        Determines whether the agent is blocked given the agent position ([x, y]).
        """
        agent_sim_position = self.convert_xy_to_xyz_in_habitat(obs)
        return not self._simulator.pathfinder.is_navigable(agent_sim_position)

    def get_xy_in_habitat(self) -> TypeHabitatXY:
        """
        Returns the position ([x, y]) of the agent in the environment.
        """
        agent_state = self._simulator.get_agent(
            self._simulator_settings["default_agent"]
        ).get_state()
        return np.array([agent_state.position[2], agent_state.position[0]])

    def set_agent_pos_w_habitatxy(self, obs: TypeHabitatXY):
        """
        Given the agent position ([x, y]), update the agent state in the environment ([y, z, x]).
        """
        agent_state = self._simulator.get_agent(
            self._simulator_settings["default_agent"]
        ).get_state()
        agent_state.position = self.convert_xy_to_xyz_in_habitat(obs)
        agent_state.sensor_states = {}
        self._agent.set_state(agent_state)

    def get_sensor_observation(self) -> TypeHabitatSensorObs:
        observations = self._simulator.get_sensor_observations()
        cat_obs = None
        if self.sensor_type == "rgb":
            cat_obs = np.zeros((4, self._height, self._width, 4), dtype=np.uint8)
        elif self.sensor_type == "depth":
            cat_obs = np.zeros((4, self._height, self._width))

        assert cat_obs is not None
        for idx, (_, value) in enumerate(observations.items()):
            cat_obs[idx] = value
        return cat_obs

    def get_bounds(self) -> Tuple[NDArray, NDArray]:
        """
        Returns the bounds of the environment.
        """
        bounds = self._simulator.pathfinder.get_bounds()
        return bounds

    def get_agent_position(self) -> TypeHabitatXYZ:
        """
        Returns the position of the agent in the environment.
        """
        agent_state = self._simulator.get_agent(
            self._simulator_settings["default_agent"]
        ).get_state()
        return np.array(agent_state.position, dtype=np.float32)


class GoalConditionedHabitatPointWrapper(gym.Wrapper):
    """Wrapper that appends goal to observation produced by habitat environment."""

    def __init__(
        self,
        env: gym.Env,
        prob_constraint: float = 0.8,
        min_dist: float = 0.0,
        max_dist: float = 4.0,
        threshold_distance: float = 1.0,
    ):
        """Initialize the environment.

        Args:
          env: an environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial observation.
          max_dist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial observation.
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """

        assert isinstance(env, HabitatNavigationEnv)
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._prob_constraint = prob_constraint
        self._threshold_distance = threshold_distance
        super(GoalConditionedHabitatPointWrapper, self).__init__(env)

        self.observation_space = Dict(
            {
                "observation": env.observation_space,
                "goal": env.observation_space,
            }
        )

        self.grid_observation_space = Dict(
            {
                "observation": env.grid_observation_space,
                "goal": env.grid_observation_space,
            }
        )

    def set_sample_goal_args(
        self,
        prob_constraint: Union[float, None] = None,
        min_dist: Union[float, None] = None,
        max_dist: Union[float, None] = None,
    ):
        assert min_dist is not None
        assert min_dist >= 0

        assert max_dist is not None
        assert max_dist >= min_dist

        assert prob_constraint is not None

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._prob_constraint = prob_constraint

    def _is_done(self, obs: NDArray, goal: NDArray) -> bool:
        """Determines whether observation equals goal."""
        return bool(np.linalg.norm(obs - goal) < self._threshold_distance)

    def _sample_goal(
        self, obs: TypeGridXY
    ) -> Tuple[TypeGridXY, Union[TypeGridXY, None]]:
        """Sampled a goal observation."""
        if np.random.random() < self._prob_constraint:
            return self._sample_goal_constrained(obs, self._min_dist, self._max_dist)
        else:
            return self._sample_goal_unconstrained(obs)

    def _sample_goal_constrained(
        self, obs: TypeGridXY, min_dist: float, max_dist: float
    ) -> Tuple[TypeGridXY, Union[TypeGridXY, None]]:
        """Samples a goal with dist min_dist <= d(observation, goal) <= max_dist.

        Args:
          obs: The current position of the agent (without goal).
          min_dist: (int) Minimum distance to goal.
          max_dist: (int) Maximum distance to goal.
        Returns:
          obs: The current position of the agent (without goal).
          goal: A goal observation that satifies the constraints.
        """
        self.env: HabitatNavigationEnv

        (i, j) = self.env._discretize_state(obs)
        mask = np.logical_and(
            self.env._apsp[i, j] >= min_dist, self.env._apsp[i, j] <= max_dist
        )
        mask = np.logical_and(mask, self.env._walls == 0)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (obs, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array(
            [candidate_states[0][goal_index], candidate_states[1][goal_index]],
            dtype=np.float32,
        )
        goal += np.random.uniform(size=2).astype(goal.dtype)
        dist_to_goal = self.get_distance(obs, goal)
        assert min_dist <= dist_to_goal <= max_dist
        assert not self.env._is_blocked(goal)
        return (obs, goal)

    def _sample_goal_unconstrained(self, obs: TypeGridXY):
        """Samples a goal without any constraints.

        Args:
          obs: observation (without goal).
        Returns:
          observation: observation (without goal).
          goal: a goal observation.
        """
        return (obs, self.env.sample_empty_state())

    def normalize_obs(self, obs: TypeGridXY):
        """get visual obs"""
        return self.get_sensor_obs_at_grid_xy(obs)

    def reset(self):
        goal = None
        count = 0
        while goal is None:
            obs = self.env.reset()
            (obs, goal) = self._sample_goal(obs)
            count += 1
            if count > 1000:
                print("WARNING: Unable to find goal within constraints.")
        self._goal = goal
        return {
            "observation": self.normalize_obs(obs),
            "goal": self.normalize_obs(self._goal),
            "grid": {
                "observation": np.copy(obs),
                "goal": np.copy(self._goal),
            },
        }

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        rew = -1.0
        done = self._is_done(obs, self._goal)
        return (
            {
                "observation": self.normalize_obs(obs),
                "goal": self.normalize_obs(self._goal),
                "grid": {
                    "observation": np.copy(obs),
                    "goal": np.copy(self._goal),
                },
            },
            rew,
            done,
            {},
        )

    @property
    def max_goal_dist(self) -> float:
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])


def plot_wall(walls: np.ndarray, ax: Axes, normalize=True):
    height, width = walls.shape
    if not normalize:
        height, width = 1.0, 1.0
    for i, j in zip(*np.where(walls)):
        x = np.array([i, i + 1]) / float(height)
        y0 = np.array([j, j]) / float(width)
        y1 = np.array([j + 1, j + 1]) / float(width)
        ax.fill_between(x, y0, y1, color="grey")
        ax.set_aspect("equal", adjustable="box")

    if normalize:
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    return ax


def plot_traj(
    traj: np.ndarray,
    walls: np.ndarray,
    normalize: bool,
    ax: Axes,
    **kwargs,
):
    height, width = walls.shape
    if normalize:
        ax.plot(traj[:, 0] / height, traj[:, 1] / width, **kwargs)
    else:
        ax.plot(traj[:, 0], traj[:, 1], **kwargs)
    return ax


def plot_start_n_goals(goals, walls: np.ndarray, ax: Axes):
    height, width = walls.shape
    ax.scatter(
        goals[:, 0] / float(height),
        goals[:, 1] / float(width),
        s=4,
        marker="*",
        c="red",
    )
    return ax


class TimeLimit(gym.Wrapper):
    """
    End episodes after specified number of steps.

    Resets the environment if either these conditions holds:
        1. The base environment returns done = True
        2. The time limit is exceeded.

    If terminate_on_timeout=True, then returns done = True in case 1 and 2
    If terminate_on_timeout=False, then returns done = True only in case 1
    """

    def __init__(self, env, duration, terminate_on_timeout=False):
        super().__init__(env)
        self.duration = duration
        self.num_steps = None
        self.terminate_on_timeout = terminate_on_timeout

    def reset(self):
        self.step_count = 0
        reset_result = self.env.reset()
        if len(reset_result) == 2:
            observation, _ = reset_result
        else:
            observation = reset_result
        observation["first_step"] = True
        return observation

    def step(self, action, num_agents=None):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            observation, reward, done, _, info = step_result
        else:
            observation, reward, done, info = step_result

        self.step_count += 1
        timed_out = self.step_count >= self.duration
        if timed_out or done:
            info["last_step"] = True
            info["timed_out"] = timed_out
            info["terminal_observation"] = observation
            done = done if not self.terminate_on_timeout else True

            if num_agents is None:
                observation = self.reset()

        return observation, reward, done, info


def habitat_env_load_fn(
    # HabitatNavigationEnv kwargs
    env_type: Literal["HabitatSim", "ReplicaCAD"],
    height: Union[float, None] = None,
    action_noise: float = 1.0,
    simulator_settings: dict = {},
    sensor_type: Literal["rgb", "depth"] = "depth",
    apsp_path: str = "",
    device: str = "cpu",
    # Wrapper kwargs
    gym_env_wrappers: Tuple[type[GoalConditionedHabitatPointWrapper], ...] = (
        GoalConditionedHabitatPointWrapper,
    ),
    wrapper_kwargs: List[dict] = [],
    # TimeLimit kwargs
    terminate_on_timeout: bool = False,
    max_episode_steps: Union[int, None] = None,
) -> gym.Env:
    """Loads the selected environment and wraps it with the specified wrappers.

    Args:
      scene: Scene path.
      height: Height at which the vertical slice of the map must be taken for the top-down map generation.
      terminate_on_timeout: Whether to set done = True when the max episode
        steps is reached.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no timestep_limit set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.

    Returns:
      An environment instance.
    """

    env = HabitatNavigationEnv(
        env_type=env_type,
        height=height,
        action_noise=action_noise,
        simulator_settings=simulator_settings,
        sensor_type=sensor_type,
        apsp_path=apsp_path,
        device=device,
    )

    for idx, wrapper in enumerate(gym_env_wrappers):
        if idx < len(wrapper_kwargs):
            env = wrapper(env, **wrapper_kwargs[idx])
        else:
            env = wrapper(env)

    if max_episode_steps is not None and max_episode_steps > 0:
        env = TimeLimit(
            env, max_episode_steps, terminate_on_timeout=terminate_on_timeout
        )
    return env


def set_habitat_env_difficulty(eval_env: gym.Env, difficulty: float):
    assert 0 <= difficulty <= 1
    max_goal_dist = eval_env.max_goal_dist  # type: ignore
    eval_env._set_sample_goal_args(  # type: ignore
        prob_constraint=1,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:X")
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        help="test scene name",
    )

    args = parser.parse_args()

    habitat_data_root = ""
    with open("configs/habitat_data.yaml", "r") as f:
        habitat_data_f = yaml.safe_load(f)
        habitat_data_root = habitat_data_f["HATBITAT_DATA_DIR"]

    env = habitat_env_load_fn(env_type='HabitatSim', height=0, device=args.device)
