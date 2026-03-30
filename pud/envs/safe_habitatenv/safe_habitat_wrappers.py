import gym
import gym.spaces
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, List, Optional

from pud.envs.habitat_navigation_env import TypeGridXY
from pud.envs.safe_pointenv.safe_wrappers import SafeTimeLimit
from pud.envs.safe_habitatenv.safe_habitatenv import SafeHabitatNavigationEnv


class SafeGoalConditionedHabitatPointWrapper(gym.Wrapper):
    """
    Wrapper that appends the goal to the observation produced by the habitat environment.
    Samples the goal with safety constraints
    Option 1: Sample goals of any distance subject to the step cost == 0
        Potential long distances, but there exists viable solutions
    Option 2: Distance constraints subject to step cost == 0
        Distance constraints, but there exists viable solutions
    Option 3: Distance constraints subject to 0 < step cost < cost limit
        Perhaps only good for training
        Distance constraints, but there may not exist viable solutions, but trajectories whose step cost < cost limit
    """

    def __init__(
        self,
        env,
        prob_constraint: float = 0.8,
        min_dist: float = 0,
        max_dist: float = 4,
        min_cost: float = 0,
        max_cost: float = 1000,
        threshold_distance: float = 1.0,
    ):
        self.env = env

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._min_cost = min_cost
        self._max_cost = max_cost
        self._prob_constraint = prob_constraint
        self._threshold_distance = threshold_distance
        super(SafeGoalConditionedHabitatPointWrapper, self).__init__(env)

        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "goal": env.observation_space,
            }
        )

    def get_prob_constraint(self):
        return self._prob_constraint

    def set_prob_constraint(self, other_pc: float):
        self._prob_constraint = other_pc

    def normalize_obs(self, obs: TypeGridXY):
        """get visual obs"""
        return self.get_sensor_obs_at_grid_xy(obs)

    def reset_orig(self):
        goal, info = None, {"cost": 0.0}
        count = 0
        while goal is None:
            obs, info = self.env.reset()  # type: ignore
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
        }, info

    def reset(self) -> Tuple[Dict, Dict]:
        return self.reset_orig()

    def _is_done(self, agent_position: NDArray, goal: NDArray) -> bool:
        """Determines whether observation equals goal.
        NOTE: Both the agent position and goal arguments are the 2D coordinates ([x, y])
        """
        return bool(np.linalg.norm(agent_position - goal) < self._threshold_distance)

    def _set_sample_goal_args(
        self,
        prob_constraint: Optional[float] = None,
        min_dist: Optional[float] = None,
        max_dist: Optional[float] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
    ):
        if prob_constraint is not None:
            self._prob_constraint = prob_constraint
        if min_dist is not None:
            self._min_dist = min_dist
        if max_dist is not None:
            self._max_dist = max_dist
        if min_cost is not None:
            self._min_cost = min_cost
        if max_cost is not None:
            self._max_cost = max_cost

    def _sample_goal(self, obs):
        """Sampled a goal observation. Use only unconstrained samples"""
        return self._sample_goal_unconstrained(obs)

    def _sample_goal_unconstrained(self, obs):
        """
        Samples a goal without any constraints.

        Args:
          obs: Observation (without goal).
        Returns:
          observation: Observation (without goal).
          goal: A goal observation.
        """
        return (obs, self.env.sample_empty_state())

    def step(self, action: NDArray) -> Tuple[Dict, float, bool, Dict]:
        obs, _, _, info = self.env.step(action)
        rew = -1.0
        done = self._is_done(obs, self._goal)
        info["success"] = done
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
            info,
        )

    @property
    def max_goal_dist(self):
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])


class SafeGoalConditionedHabitatPointQueueWrapper(
    SafeGoalConditionedHabitatPointWrapper
):
    def __init__(
        self,
        env: SafeGoalConditionedHabitatPointWrapper,
        prob_constraint: float = 0.8,
        min_dist=0,
        max_dist=4,
        min_cost=0,
        max_cost=1000,
        threshold_distance=1.0,
    ):
        """
        Reset using problems (start-goal pairs) from an external queue.
        If the queue is empty, use the default reset method
        """
        super(SafeGoalConditionedHabitatPointQueueWrapper, self).__init__(
            env=env,
            prob_constraint=prob_constraint,
            min_dist=min_dist,
            max_dist=max_dist,
            min_cost=min_cost,
            max_cost=max_cost,
            threshold_distance=threshold_distance,
        )
        self.pb_Q = []
        # By default, don't pop from Q because there are many
        # Redundant reset from parent classes
        self.use_q = False
        self.verbose = True

    def get_Q_size(self):
        return len(self.pb_Q)

    def set_use_q(self, status: bool):
        self.use_q = status

    def append_pbs(self, pb_list: List[tuple]):
        self.pb_Q.extend(pb_list)

    def set_pbs(self, pb_list: List[tuple]):
        """replace the problem Q with a new one,
        intended for update pbs for training"""
        assert isinstance(pb_list, list)
        self.pb_Q = pb_list

    def set_verbose(self, new_verbose: bool):
        self.verbose = new_verbose

    def reset(self):
        if self.use_q and np.random.rand() < self._prob_constraint:
            if len(self.pb_Q) > 0:
                new_pb = self.pb_Q.pop(0)
                return self.reset_alt(**new_pb)  # type: ignore
            if self.verbose:
                print("[WARN]: queue from goal conditioned env is empty")
        return self.reset_orig()

    def reset_alt(self, start: np.ndarray, goal: np.ndarray, info: dict = {}):
        """reset using alternative source, start and goal are assumed to be de-normalized"""
        self._goal = goal
        obs, new_info = self.env.reset_manual(start_state=start)
        new_info.update(info)
        return {
            "observation": self.normalize_obs(obs),
            "goal": self.normalize_obs(self._goal),
            "grid": {
                "observation": np.copy(obs),
                "goal": np.copy(self._goal),
            },
        }, new_info


def set_safe_habitat_env_difficulty(
    eval_env: SafeGoalConditionedHabitatPointWrapper,
    difficulty: float,
    min_cost: float = 0.0,
    max_cost: float = 1.0,
):

    assert 0 <= difficulty <= 1

    max_goal_dist = eval_env.max_goal_dist
    eval_env._set_sample_goal_args(
        prob_constraint=1,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05),
        min_cost=min_cost,
        max_cost=max_cost,
    )


def safe_habitat_env_load_fn(
    env_kwargs: dict,
    cost_f_args: dict,
    cost_limit: float,
    max_episode_steps: int = 0,
    gym_env_wrappers: Tuple[gym.Wrapper] = (
        SafeGoalConditionedHabitatPointWrapper,
    ),  # type: ignore
    wrapper_kwargs: List[dict] = [],
    terminate_on_timeout=False,
):
    """
    Loads the selected environment and wraps it with the specified wrappers.

    Args:
      environment_name: Name for the environment to load.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no timestep_limit set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      wrapper_kwargs: args for gym_env_wrappers, empty list or [wrapper_1_arg, wrapper_2_arg, ...],
        where wrapper_N_arg could be empty tuple as a place holder
      terminate_on_timeout: Whether to set done = True when the max episode
                            steps is reached.

    Returns:
      An environment instance.
    """

    env = SafeHabitatNavigationEnv(
        **env_kwargs,
        cost_f_args=cost_f_args,
        cost_limit=cost_limit,
    )

    for idx, wrapper in enumerate(gym_env_wrappers):
        if idx < len(wrapper_kwargs):
            env = wrapper(env, **wrapper_kwargs[idx])  # type: ignore
        else:
            env = wrapper(env)  # type: ignore

    if max_episode_steps is not None and max_episode_steps > 0:
        env = SafeTimeLimit(
            env, max_episode_steps, terminate_on_timeout=terminate_on_timeout
        )
    return env
