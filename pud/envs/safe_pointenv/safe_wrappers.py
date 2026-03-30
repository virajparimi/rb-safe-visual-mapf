import gym
import pickle
import gym.spaces
import numpy as np
from typing import List, Union
from numpy.typing import NDArray

from pud.envs.wrappers import TimeLimit
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.algos.cbfs.cbfs_eval import sample_precompiled_grid_policies


class SafeGoalConditionedPointWrapper(gym.Wrapper):
    """
    Wrapper that appends goal to observation produced by environment.
    Sample with safety constraints
    Option 1: Sample goals of any distance subject to step cost == 0
        Potential long distances, but there exists viable solutions
    Option 2: Distance constraints subject to step cost == 0
        Distance constraints, but there exists viable solutions
    Option 3: Distance constraints subject to 0 < step cost < cost limit
        Perhaps only good for training
        Distance constraints, but there may not exist viable solutions, but trajectories whose step cost < cost limit
    """

    def __init__(
        self,
        env: SafePointEnv,
        prob_constraint: float = 0.8,
        min_dist=0,
        max_dist=4,
        min_cost=0,
        max_cost=1000,
        threshold_distance=1.0,
    ):
        """Initialize the environment.

        Args:
          env: An environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial observation.
          max_dist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial observation.
          reset_blend: (float) the probability of running the reset with balanced sampling,
                       0 means only running the original reset, 1 means running only the balanced reset
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._min_cost = min_cost
        self._max_cost = max_cost
        self._prob_constraint = prob_constraint
        self._threshold_distance = threshold_distance
        super(SafeGoalConditionedPointWrapper, self).__init__(env)

        # Make sure to use gym, not gymnasium
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "goal": env.observation_space,
            }
        )
        # Load CBFS sample policies on grid
        self.env: SafePointEnv  # for auto-complete

    def get_prob_constraint(self):
        return self._prob_constraint

    def set_prob_constraint(self, other_pc: float):
        self._prob_constraint = other_pc

    def normalize_obs(self, obs):
        return np.array(
            [obs[0] / float(self.env._height), obs[1] / float(self.env._width)],
            dtype=self.observation_space["observation"].dtype,  # type: ignore
        )

    def de_normalize_obs(self, obs: np.ndarray):
        """reverse of _normalize_obs"""
        return np.array(
            [
                obs[0] * float(self.env._height),
                obs[1] * float(self.env._width),
            ],
            dtype=self.observation_space["observation"].dtype,  # type: ignore
        )

    def de_normalize_goal_conditioned_obs(self, obs: dict):
        """reverse of _normalize_obs"""
        out = {
            "observation": self.de_normalize_obs(obs["observation"]),
            "goal": self.de_normalize_obs(obs["goal"]),
        }
        return out

    def step(self, action):
        """
        The safe_pointenv does NOT use normalized observations, the goal-conditioned env does
        Make sure the cost is computed from the safe_pointenv using the un-normalized observations

        NOTE: The step is still computed by safe_pointenv, so the internal variables are all un-normalized
        """
        obs, _, _, info = self.env.step(action)
        rew = -1.0
        done = bool(self._is_done(obs, self._goal))
        info["success"] = done
        return (
            {
                "observation": self.normalize_obs(obs),
                "goal": self.normalize_obs(self._goal),
            },
            rew,
            done,
            info,
        )

    def set_sample_goal_args(
        self,
        prob_constraint=None,
        min_dist=None,
        max_dist=None,
        min_cost=None,
        max_cost=None,
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

    def _is_done(self, obs, goal):
        """
        Determines whether observation equals goal
        """
        return np.linalg.norm(obs - goal) < self._threshold_distance

    def reset(self):
        return self.reset_orig()

    #########################################
    # original sampling functions, verified to work well on policy training
    #########################################
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
        }, info

    def _sample_goal(self, obs):
        """Sampled a goal observation.
        use only unconstrained samples"""
        return self._sample_goal_unconstrained(obs)

    def _sample_goal_constrained(self, obs, min_dist, max_dist):
        """
        Samples a goal with distance min_dist <= d(observation, goal) <= max_dist.

        Args:
          obs: Observation (without goal).
          min_dist: (int) Minimum distance to goal.
          max_dist: (int) Maximum distance to goal.

        Returns:
          observation: Observation (without goal).
          goal: A goal observation.
        """

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

        goal += np.random.uniform(size=2)
        dist_to_goal = self.env._get_distance(obs, goal)

        assert min_dist <= dist_to_goal <= max_dist
        assert not self.env._is_blocked(goal)
        return (obs, goal)

    def _sample_goal_unconstrained(self, obs):
        """
        Samples a goal without any constraints.

        Args:
          obs: Observation (without goal).
        Returns:
          observation: Observation (without goal).
          goal: A goal observation.
        """
        return (obs, self.env._sample_empty_state())

    @property
    def max_goal_dist(self):
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])


class SafeGoalConditionedPointQueueWrapper(SafeGoalConditionedPointWrapper):
    def __init__(
        self,
        env: SafePointEnv,
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
        super(SafeGoalConditionedPointQueueWrapper, self).__init__(
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
        """
        Replace the problem Q with a new one, intended for update pbs for training
        """
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

    def reset_alt(self, start: NDArray, goal: NDArray, info: dict = {}):
        """reset using alternative source, start and goal are assumed to be de-normalized"""
        self._goal = goal
        obs, new_info = self.env.reset_manual(start_state=start)
        new_info.update(info)
        return {
            "observation": self.normalize_obs(obs),
            "goal": self.normalize_obs(self._goal),
        }, new_info


class SafeGoalConditionedPointBlendWrapper(SafeGoalConditionedPointWrapper):
    def __init__(
        self,
        env: SafePointEnv,
        prob_constraint: float = 0.8,
        min_dist=0,
        max_dist=4,
        min_cost=0,
        max_cost=1000,
        reset_blend=1.0,
        threshold_distance=1.0,
        cbfs_policy_path: str = "",  # path to pre-compiled sample policies on grid
    ):
        """Balance the expected accumulated costs by blending reset with a precompiled sample policy on grid."""
        self.reset_blend = reset_blend
        super(SafeGoalConditionedPointBlendWrapper, self).__init__(
            env=env,
            prob_constraint=prob_constraint,
            min_dist=min_dist,
            max_dist=max_dist,
            min_cost=min_cost,
            max_cost=max_cost,
            threshold_distance=threshold_distance,
        )

        # Load CBFS sample policies on grid
        self.load_cbfs_grid_policy(cbfs_policy_path)

        self.env: SafePointEnv

    def load_cbfs_grid_policy(self, file_path):
        if (not hasattr(self, "pi_cbfs")) or (self.pi_cbfs is None):
            self.pi_cbfs = None
            with open(file_path, "rb") as f:
                self.pi_cbfs = pickle.load(f)
        return

    def cbfs_sample(
        self,
        min_dist: float,
        max_dist: float,
        min_cost: float,
        max_cost: float,
        max_attempts: int = 100,
    ):
        """
        Sampling start and goal states with guaranteed solution with distance constraints
        """
        assert hasattr(self, "pi_cbfs"), "cbfs grid policy not loaded"

        for _ in range(max_attempts):
            out = sample_precompiled_grid_policies(
                self.pi_cbfs,  # type: ignore
                min_cost=min_cost,
                max_cost=max_cost,
                min_len=min_dist,
                max_len=max_dist,
            )
            if out:
                traj, traj_cost = out
                return traj, traj_cost
        raise Exception(
            "Failed to generate a valid grid trajectory sample for spec:"
            "min_dist={}, max_dist={}, min_cost={}, max_cost={}, max_attempts={}".format(
                min_dist, max_dist, min_cost, max_cost, max_attempts
            )
        )

    def reset(self):  # type: ignore
        if np.random.random() < self.reset_blend:
            return self.reset_cost()
        else:
            return self.reset_orig()

    def reset_cost(self):
        """
        P(prob_constraint): sample under length and cost constraint
        P(1-prob_constraint): sample with no constraints
        """
        out = dict()
        if np.random.random() < self._prob_constraint:
            traj, traj_cost = self.cbfs_sample(
                min_dist=self._min_dist,
                max_dist=self._max_dist,
                min_cost=self._min_cost,
                max_cost=self._max_cost,
            )
            out["s0"], out["sg"] = np.array(traj[0], dtype=float), np.array(
                traj[-1], dtype=float
            )
        else:
            obs, info = self.env.reset()  # type: ignore
            (out["s0"], out["sg"]) = self._sample_goal_unconstrained(obs=obs)
        self._goal = out["sg"]
        obs = out["s0"]
        self.state = obs.copy()
        cost = self.env.get_state_cost(self._goal)

        new_state = {
            "observation": self.normalize_obs(obs),
            "goal": self.normalize_obs(self._goal),
        }
        info = {"cost": cost}
        return new_state, info


class SafeTimeLimit(TimeLimit):
    def __init__(self, env, duration, terminate_on_timeout=False):
        super(SafeTimeLimit, self).__init__(
            env=env,
            duration=duration,
            terminate_on_timeout=terminate_on_timeout,
        )

    def step(self, action, num_agents=None):
        observation, reward, done, info = super(SafeTimeLimit, self).step(
            action, num_agents=num_agents
        )
        new_obs = observation
        if isinstance(observation, tuple):
            # A reset happens, separate the obs and info
            # store the first step info in the new obs
            # discard the last obs
            new_obs, new_info = observation
            new_obs["first_info"] = new_info
        return new_obs, reward, done, info

    def reset(self):
        """Reset adds a info dict"""
        self.step_count = 0
        observation, info = self.env.reset()
        observation["first_step"] = True
        return observation, info


def set_safe_env_difficulty(
    eval_env: Union[SafeTimeLimit, SafeGoalConditionedPointWrapper],
    difficulty: float,
    min_cost: float = 0.0,
    max_cost: float = 1.0,
):
    assert 0 <= difficulty <= 1
    max_goal_dist = eval_env.max_goal_dist
    eval_env.set_sample_goal_args(
        prob_constraint=1,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05),
        min_cost=min_cost,
        max_cost=max_cost,
    )


def safe_env_load_fn(
    env_kwargs: dict,
    cost_f_kwargs: dict,
    max_episode_steps=0,
    gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
    wrapper_kwargs: List[dict] = [],
    terminate_on_timeout=False,
):
    """Loads the selected environment and wraps it with the specified wrappers.

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
    env = SafePointEnv(**env_kwargs, cost_f_args=cost_f_kwargs)

    for idx, wrapper in enumerate(gym_env_wrappers):
        if idx < len(wrapper_kwargs):
            env = wrapper(env, **wrapper_kwargs[idx])
        else:
            env = wrapper(env)

    if max_episode_steps is not None and max_episode_steps > 0:
        env = SafeTimeLimit(
            env, max_episode_steps, terminate_on_timeout=terminate_on_timeout
        )

    return env
