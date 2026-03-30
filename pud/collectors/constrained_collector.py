from __future__ import annotations

import logging
import numpy as np
from typing import Union

from pud.collectors.collector import Collector
from pud.algos.policies import BasePolicy, GaussianPolicy
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer


def eval_agent_from_Q(policy, eval_env, collect_trajs=False):
    """
    Run evaluation and records the initial states for each episode
    until the pb Q from the env is empty. At the end of the last pb in the Q, the step will trigger another
    reset, and it should be safely handled by the reset_orig, suppress the warning
    message by turning off verbose
    """
    # Verify the eval_env has an non-empty Q of pbs
    assert hasattr(eval_env, "pb_Q")
    bk_prob_constriant = eval_env.get_prob_constraint()
    eval_env.set_prob_constraint(1.0)  # Only use from the pb Q
    eval_env.set_verbose(False)
    eval_env.set_use_q(True)

    records = {}

    def new_record(init_state: Union[np.ndarray, dict], info: dict = {}):
        key = len(records.keys())
        records[key] = {
            "rewards": 0.0,
            "cum_costs": info["cost"],
            "max_step_cost": 0.0,
            "steps": 0,
            "init_info": info,
        }
        if collect_trajs:
            records[key]["traj"] = [eval_env.get_internal_state()]
        records[key]["init_states"] = eval_env.de_normalize_goal_conditioned_obs(
            init_state
        )
        return key

    c = 0
    n = eval_env.get_Q_size()

    if n == 0:
        return records

    state, info = eval_env.reset()
    cur_key = new_record(state, info)

    while c < n:
        action = policy.select_action(state)
        """When episode ends:
            - state is the new state of the new epsiode
            - reward, done, info are from the last step of the terminated epsiode
        """
        state, reward, done, info = eval_env.step(np.copy(action))

        records[cur_key]["steps"] += 1
        records[cur_key]["rewards"] += reward

        if (not done) and collect_trajs:
            records[cur_key]["traj"].append(eval_env.get_internal_state())

        co = info.get("cost", 0.0)
        if co > records[cur_key]["max_step_cost"]:
            records[cur_key]["max_step_cost"] = co
        records[cur_key]["cum_costs"] += co

        if done:
            records[cur_key]["success"] = info["success"]
            if collect_trajs and "terminal_observation" in info:
                records[cur_key]["traj"].append(
                    eval_env.de_normalize_obs(
                        info["terminal_observation"]["observation"]
                    )
                )

            c += 1
            if c < n:
                cur_key = new_record(state, state["first_info"])
                assert state["first_step"]
            else:
                eval_env.set_use_q(False)

    eval_env.set_verbose(True)
    eval_env.set_prob_constraint(bk_prob_constriant)
    return records


class ConstrainedCollector(Collector):
    def __init__(
        self,
        policy: GaussianPolicy,
        buffer: ConstrainedReplayBuffer,
        env,
        initial_collect_steps: int = 0,
    ):
        super(ConstrainedCollector, self).__init__(
            policy=policy,
            buffer=buffer,
            env=env,
            initial_collect_steps=initial_collect_steps,
        )
        assert isinstance(
            self.buffer, ConstrainedReplayBuffer
        ), "Error: Need to use ConstrainedReplayBuffer"

        self.past_eps = []
        self.num_eps = 0
        self.state, _ = env.reset()
        self._reset_log()

    def _reset_log(self) -> None:
        """
        Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        self._ep_ret = 0.0
        self._ep_cost = 0.0
        self._ep_len = 0.0
        self._ep_cost_max = 0.0

    def _append_ep_log(self):
        self.past_eps.append(
            {
                "ep_ret": self._ep_ret,
                "ep_cost": self._ep_cost,
                "ep_len": self._ep_len,
                "ep_cost_max": self._ep_cost_max,
            }
        )
        self.num_eps += 1

    def _log_metric(self, reward: float, cost: float):
        self._ep_len += 1
        self._ep_ret += reward
        self._ep_cost += cost
        if self._ep_cost_max < cost:
            self._ep_cost_max = cost

    def step(self, num_steps):
        """
        Step num_steps in the env.
        NOTE: The env is not reset before stepping, the env is kept alive after exiting this method
        """
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)

            next_state, reward, done, info = self.env.step(np.copy(action))
            self._log_metric(reward, cost=info["cost"])

            if info.get("last_step", False):
                self.buffer.add(
                    self.state,
                    action,
                    info["terminal_observation"],
                    reward,
                    info["cost"],
                    done,
                )
                self._append_ep_log()
                self._reset_log()
            else:
                self.buffer.add(
                    self.state, action, next_state, reward, info["cost"], done
                )

            self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            s0, _ = eval_env.reset()
            rb_vec.append(s0)
        rb_vec = np.array([x["observation"] for x in rb_vec])
        return rb_vec

    @classmethod
    def sample_initial_unconstrained_grid_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            obs = eval_env.sample_empty_state()
            s0 = eval_env.normalize_obs(obs)
            rb_vec.append(s0)
        rb_vec = np.array(rb_vec)
        return rb_vec

    @classmethod
    def _get_policy_cbs_plan_cost(cls, policy):
        getter = getattr(policy, "get_last_cbs_plan_risk", None)
        if callable(getter):
            return getter()
        return None

    @classmethod
    def _get_policy_cbs_plan_time(cls, policy):
        getter = getattr(policy, "get_last_cbs_plan_time", None)
        if callable(getter):
            return getter()
        return None

    @classmethod
    def sample_initial_unconstrained_visual_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            state, _ = eval_env.reset()
            rb_vec.append(state)
        rb_vec_grid = np.array([x["grid"]["observation"] for x in rb_vec])
        rb_vec_visual = np.array([x["observation"] for x in rb_vec])
        return rb_vec_grid, rb_vec_visual

    @classmethod
    def sample_initial_unconstrained_states(cls, eval_env, num_states, habitat=False):
        if habitat:
            return cls.sample_initial_unconstrained_visual_states(eval_env, num_states)
        else:
            return cls.sample_initial_unconstrained_grid_states(eval_env, num_states)

    @classmethod
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """
        c = 0
        r = 0
        rewards = []

        co = 0
        cum_co = 0
        max_co = 0
        max_costs = []
        cum_costs = []
        state = eval_env.reset()
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode:
                c += 1

            r += reward

            co = info.get("cost", 0.0)
            if co > max_co:
                max_co = co
            cum_co += co

            if done:
                rewards.append(r)
                cum_costs.append(cum_co)
                max_costs.append(max_co)
                if by_episode:
                    c += 1
                r = 0

                co = 0
                cum_co = 0
                max_co = 0

        eval_outputs = {
            "returns": rewards,
            "max_costs": max_costs,
            "cum_costs": cum_costs,
        }
        return eval_outputs

    @classmethod
    def eval_agent_n_record_init_states(cls, policy, eval_env, n, by_episode=True):
        """
        Run evaluation and records the initial states for each episode
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        records = {}

        def new_record(init_state: Union[np.ndarray, dict]):
            key = len(records.keys())
            records[key] = {
                "rewards": 0.0,
                "costs": 0.0,
                "max_step_cost": 0.0,
                "init_states": init_state,
                "steps": 0,
            }
            return key

        c = 0
        r, co, max_co, cum_co = [0.0] * 4
        state, info = eval_env.reset()
        cur_key = new_record(state)
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode:
                c += 1
            records[cur_key]["steps"] += 1

            r += reward

            co = info.get("cost", 0.0)
            if co > max_co:
                max_co = co
            cum_co += co

            if done:
                records[cur_key]["rewards"] = r
                records[cur_key]["costs"] = co
                records[cur_key]["max_step_cost"] = max_co
                if by_episode:
                    c += 1
                    if c < n:
                        cur_key = new_record(state)
                        assert state["first_step"]

                r, co, max_co, cum_co = [0.0] * 4
        return records

    @classmethod
    def step_cleanup(cls, search_policy, eval_env, num_steps):
        # cls stands for class
        c = 0
        while c < num_steps:
            goal = search_policy.get_goal_in_rb()
            state, _ = eval_env.reset()
            done = False

            while True:
                state["goal"] = goal
                try:
                    action = search_policy.select_action(state)
                except Exception as e:
                    raise e

                state, reward, done, info = eval_env.step(np.copy(action))
                c += 1

                if done or c >= num_steps or search_policy.reached_final_waypoint:
                    break

    @classmethod
    def get_grid_trajectory(
        cls, policy, eval_env, start=None, goal=None, start_cost=None
    ):
        ep_reward_list = []
        ep_waypoint_list = []
        ep_observation_list = []

        state, info = eval_env.reset()
        start_cost_value = info["cost"] if start_cost is None else start_cost
        denormalize_factor = np.array(
            [eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32
        )

        if start is not None and goal is not None and start_cost is not None:
            state["goal"] = goal.copy()
            state["observation"] = start.copy()
            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goal * denormalize_factor
            eval_env.unwrapped.state = start * denormalize_factor

        ep_goal = state["goal"]
        ep_start = state["observation"]
        ep_record = {
            "steps": 0,
            "rewards": 0.0,
            "max_step_cost": 0.0,
            "first_step_cost": start_cost_value,
            "cumulative_costs": start_cost_value,
            "cbs_plan_cost": None,
            "cbs_plan_time": None,
        }

        while True:
            ep_observation_list.append(state["observation"])

            # NOTE: state['goal'] may be modified
            action = policy.select_action(state)
            if ep_record["cbs_plan_cost"] is None:
                plan_cost = cls._get_policy_cbs_plan_cost(policy)
                if plan_cost is not None:
                    ep_record["cbs_plan_cost"] = plan_cost
            if ep_record["cbs_plan_time"] is None:
                plan_time = cls._get_policy_cbs_plan_time(policy)
                if plan_time is not None:
                    ep_record["cbs_plan_time"] = plan_time

            ep_waypoint_list.append(state["goal"])

            state, reward, done, info = eval_env.step(np.copy(action), num_agents=1)

            ep_record["steps"] += 1
            ep_record["rewards"] += reward

            cost = info.get("cost", 0.0)
            if cost > ep_record["max_step_cost"]:
                ep_record["max_step_cost"] = cost
            ep_record["cumulative_costs"] += cost

            ep_reward_list.append(reward)
            if done:
                ep_record["success"] = info["success"]
                ep_observation_list.append(info["terminal_observation"]["observation"])
                break

        return (
            ep_start,
            ep_goal,
            ep_observation_list,
            ep_waypoint_list,
            ep_reward_list,
            ep_record,
        )

    @classmethod
    def get_visual_trajectory(
        cls, policy, eval_env, start=None, goal=None, start_cost=None
    ):
        ep_reward_list = []
        ep_waypoint_list = []
        ep_observation_list = []

        state, info = eval_env.reset()
        start_cost_value = info["cost"] if start_cost is None else start_cost

        if start is not None and goal is not None:
            state["goal"] = goal[1].copy()
            state["grid"]["goal"] = goal[0].copy()
            state["observation"] = start[1].copy()
            state["grid"]["observation"] = start[0].copy()
            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goal[0]
            eval_env.unwrapped.state_grid = state["grid"]["observation"]

        ep_goal = (state["grid"]["goal"], state["goal"])
        ep_start = (state["grid"]["observation"], state["observation"])
        ep_record = {
            "steps": 0,
            "rewards": 0.0,
            "max_step_cost": 0.0,
            "first_step_cost": start_cost_value,
            "cumulative_costs": start_cost_value,
            "cbs_plan_cost": None,
            "cbs_plan_time": None,
        }
        while True:
            ep_observation = (state["grid"]["observation"], state["observation"])
            ep_observation_list.append(ep_observation)

            action = policy.select_action(state)  # NOTE: state['goal'] may be modified
            if ep_record["cbs_plan_cost"] is None:
                plan_cost = cls._get_policy_cbs_plan_cost(policy)
                if plan_cost is not None:
                    ep_record["cbs_plan_cost"] = plan_cost
            if ep_record["cbs_plan_time"] is None:
                plan_time = cls._get_policy_cbs_plan_time(policy)
                if plan_time is not None:
                    ep_record["cbs_plan_time"] = plan_time

            ep_waypoint = (state["grid"]["goal"], state["goal"])
            ep_waypoint_list.append(ep_waypoint)

            state, reward, done, info = eval_env.step(np.copy(action))
            ep_record["steps"] += 1
            ep_record["rewards"] += reward

            cost = info.get("cost", 0.0)
            if cost > ep_record["max_step_cost"]:
                ep_record["max_step_cost"] = cost
            ep_record["cumulative_costs"] += cost

            ep_reward_list.append(reward)

            if done:
                ep_record["success"] = info["success"]
                ep_observation = (
                    info["terminal_observation"]["grid"]["observation"],
                    info["terminal_observation"]["observation"],
                )
                ep_observation_list.append(ep_observation)
                break

        return (
            ep_start,
            ep_goal,
            ep_observation_list,
            ep_waypoint_list,
            ep_reward_list,
            ep_record,
        )

    @classmethod
    def get_trajectory(
        cls,
        policy,
        eval_env,
        input_start=None,
        input_goal=None,
        start_cost=None,
        habitat=False,
    ):
        if habitat:
            if (
                input_start is not None
                and input_goal is not None
                and start_cost is not None
            ):
                assert len(input_start) == 2 and len(input_goal) == 2
            return cls.get_visual_trajectory(
                policy, eval_env, input_start, input_goal, start_cost
            )
        else:
            return cls.get_grid_trajectory(
                policy, eval_env, input_start, input_goal, start_cost
            )

    @classmethod
    def get_grid_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        starts=None,
        goals=None,
        start_costs=None,
        threshold=0.05,
        wait=False,
    ):
        augmented_ep_reward_list = []
        augmented_ep_record_list = []
        augmented_ep_waypoint_list = []
        augmented_ep_observation_list = []
        for _ in range(num_agents):
            augmented_ep_reward_list.append([])
            augmented_ep_waypoint_list.append([])
            augmented_ep_observation_list.append([])
            augmented_ep_record_list.append(
                {
                    "steps": 0,
                    "rewards": 0.0,
                    "max_step_cost": 0.0,
                    "first_step_cost": 0.0,
                    "cumulative_costs": 0.0,
                    "cbs_plan_cost": None,
                    "cbs_plan_time": None,
                }
            )

        denormalize_factor = np.array(
            [eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32
        )

        state, info = eval_env.reset()

        if starts is not None and goals is not None and start_costs is not None:

            state["goal"] = goals.copy()
            state["observation"] = starts.copy()

            state["composite_goals"] = goals.copy()
            state["agent_waypoints"] = goals.copy()

            state["composite_starts"] = starts.copy()
            state["agent_observations"] = starts.copy()

            for agent_id in range(num_agents):
                augmented_ep_record_list[agent_id]["first_step_cost"] = start_costs[
                    agent_id
                ]
                augmented_ep_record_list[agent_id]["cumulative_costs"] = start_costs[
                    agent_id
                ]
        else:

            # Use the sampled start and goal for the first agent
            agent_goal = [state["goal"]]
            agent_start = [state["observation"]]
            augmented_ep_record_list[0]["first_step_cost"] = info.get("cost", 0.0)

            # Mutable objects
            state["agent_waypoints"] = agent_goal.copy()
            state["agent_observations"] = agent_start.copy()

            goals = agent_goal.copy()
            starts = agent_start.copy()

            # Sample the starts and goals for the other agents
            for _ in range(num_agents - 1):

                agent_state, info = eval_env.reset()
                agent_goal = [agent_state["goal"]]
                agent_start = [agent_state["observation"]]
                augmented_ep_record_list[_ + 1]["first_step_cost"] = info.get(
                    "cost", 0.0
                )

                # Add the new observations and goals to the state
                goals.extend(agent_goal.copy())
                starts.extend(agent_start.copy())
                state["agent_waypoints"].extend(agent_goal.copy())
                state["agent_observations"].extend(agent_start.copy())

            # Immutable objects - Should not change ever!
            state["composite_goals"] = goals.copy()
            state["composite_starts"] = starts.copy()
            logging.debug("Sampled the required starts and goals")

        all_done = False
        collided = [False for _ in range(num_agents)]
        agent_done = [False for _ in range(num_agents)]

        while not all_done:

            state["goal"] = state["agent_waypoints"][0]
            state["observation"] = state["agent_observations"][0]

            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goals[0] * denormalize_factor
            eval_env.unwrapped.state = state["observation"] * denormalize_factor

            # NOTE: state's agent_observations, agent_waypoints and goal are updated
            if isinstance(policy, BasePolicy):
                actions, agent_goals = policy.select_action(state)
            plan_cost = cls._get_policy_cbs_plan_cost(policy)
            if plan_cost is not None:
                for record in augmented_ep_record_list:
                    if record.get("cbs_plan_cost") is None:
                        record["cbs_plan_cost"] = plan_cost
            plan_time = cls._get_policy_cbs_plan_time(policy)
            if plan_time is not None:
                for record in augmented_ep_record_list:
                    if record.get("cbs_plan_time") is None:
                        record["cbs_plan_time"] = plan_time

            wait_agents = []
            for agent_id in range(num_agents):

                if agent_done[agent_id]:
                    continue

                if wait:
                    if agent_id in wait_agents:
                        continue

                    for other_agent_id in range(agent_id + 1, num_agents):
                        agent_state = np.array(state["agent_observations"][agent_id])
                        agent_state *= denormalize_factor
                        other_agent_state = np.array(
                            state["agent_observations"][other_agent_id]
                        )
                        other_agent_state *= denormalize_factor

                        if np.linalg.norm(agent_state - other_agent_state) < 1.0:
                            # Wait if the distance is less than MAX_ACTION
                            wait_agents.append(other_agent_id)

                if isinstance(policy, BasePolicy):
                    state["agent_waypoints"][agent_id] = agent_goals[agent_id]

                current_agent_waypoint = state["agent_waypoints"][agent_id]
                current_agent_observation = state["agent_observations"][agent_id]

                augmented_ep_waypoint_list[agent_id].append(current_agent_waypoint)
                augmented_ep_observation_list[agent_id].append(
                    current_agent_observation
                )

                state_copy = state.copy()

                state["goal"] = state["agent_waypoints"][agent_id]
                state["observation"] = state["agent_observations"][agent_id]

                action = (
                    actions[agent_id]
                    if isinstance(policy, BasePolicy)
                    else policy.select_action(state)
                )

                if "goalconditioned" in type(eval_env.env).__name__.lower():
                    eval_env.env._goal = goals[agent_id] * denormalize_factor
                eval_env.unwrapped.state = state["observation"] * denormalize_factor

                state, reward, done, info = eval_env.step(
                    np.copy(action), num_agents=num_agents
                )

                augmented_ep_record_list[agent_id]["steps"] += 1
                augmented_ep_record_list[agent_id]["rewards"] += reward

                agent_cost = info.get("cost", 0.0)
                if agent_cost > augmented_ep_record_list[agent_id]["max_step_cost"]:
                    augmented_ep_record_list[agent_id]["max_step_cost"] = agent_cost
                augmented_ep_record_list[agent_id]["cumulative_costs"] += agent_cost

                # At this point the state is changed and does not have the extra attributes so add them back
                state["composite_goals"] = state_copy["composite_goals"]
                state["composite_starts"] = state_copy["composite_starts"]

                state["agent_waypoints"] = state_copy["agent_waypoints"]
                state["agent_observations"] = state_copy["agent_observations"]

                # The agent's observations are updated based on the step function
                state["agent_observations"][agent_id] = state["observation"]

                augmented_ep_reward_list[agent_id].append(reward)

                if done:
                    augmented_ep_record_list[agent_id]["success"] = info["success"]
                    terminal_agent_observation = info["terminal_observation"][
                        "observation"
                    ]
                    augmented_ep_observation_list[agent_id].append(
                        terminal_agent_observation
                    )
                    agent_done[agent_id] = True

            # Check if any of the agent's positions are within some threshold
            for agent_id in range(num_agents):
                for other_agent_id in range(num_agents):
                    if (
                        agent_id == other_agent_id
                        or other_agent_id in wait_agents
                        or agent_done[agent_id]
                        or agent_done[other_agent_id]
                    ):
                        continue

                    agent_state = np.array(state["agent_observations"][agent_id])
                    other_agent_state = np.array(
                        state["agent_observations"][other_agent_id]
                    )

                    if np.linalg.norm(agent_state - other_agent_state) < threshold:
                        collided[agent_id] = True
                        collided[other_agent_id] = True
                        augmented_ep_record_list[agent_id]["success"] = False

            all_done = all(agent_done)

        for agent_id in range(num_agents):
            if collided[agent_id]:
                logging.info(f"Agent {agent_id} was within threshold of another agent")
                augmented_ep_record_list[agent_id]["success"] = False
                break

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
            augmented_ep_record_list,
        )

    @classmethod
    def get_visual_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        starts=None,
        goals=None,
        start_costs=None,
        threshold=0.05,
        wait=False,
    ):
        augmented_ep_reward_list = []
        augmented_ep_record_list = []
        augmented_ep_waypoint_list = []
        augmented_ep_observation_list = []
        for _ in range(num_agents):
            augmented_ep_reward_list.append([])
            augmented_ep_waypoint_list.append([])
            augmented_ep_observation_list.append([])
            augmented_ep_record_list.append(
                {
                    "steps": 0,
                    "rewards": 0.0,
                    "max_step_cost": 0.0,
                    "first_step_cost": 0.0,
                    "cumulative_costs": 0.0,
                    "cbs_plan_cost": None,
                    "cbs_plan_time": None,
                }
            )

        state, info = eval_env.reset()

        if starts is not None and goals is not None and start_costs is not None:

            assert isinstance(starts, list) and isinstance(goals, list)
            assert len(starts[0]) == 2 and len(goals[0]) == 2

            state["goal"] = goals[0][1].copy()
            state["grid"]["goal"] = goals[0][0].copy()
            state["observation"] = starts[0][1].copy()
            state["grid"]["observation"] = starts[0][0].copy()

            state["composite_goals"] = goals.copy()
            state["agent_waypoints"] = goals.copy()

            state["composite_starts"] = starts.copy()
            state["agent_observations"] = starts.copy()

            for agent_id in range(num_agents):
                augmented_ep_record_list[agent_id]["first_step_cost"] = start_costs[
                    agent_id
                ]
                augmented_ep_record_list[agent_id]["cumulative_costs"] = start_costs[
                    agent_id
                ]
        else:
            # Use the sampled start and goal for the first agent
            agent_goal = [(state["grid"]["goal"], state["goal"])]
            agent_start = [(state["grid"]["observation"], state["observation"])]
            augmented_ep_record_list[0]["first_step_cost"] = info.get("cost", 0.0)

            # Mutable objects
            state["agent_waypoints"] = agent_goal.copy()
            state["agent_observations"] = agent_start.copy()

            goals = agent_goal.copy()
            starts = agent_start.copy()

            # Sample the starts and goals for the other agents
            for _ in range(num_agents - 1):

                agent_state, info = eval_env.reset()
                agent_goal = [(agent_state["grid"]["goal"], agent_state["goal"])]
                agent_start = [
                    (agent_state["grid"]["observation"], agent_state["observation"])
                ]
                augmented_ep_record_list[_ + 1]["first_step_cost"] = info.get(
                    "cost", 0.0
                )

                # Add the new observations and goals to the state
                goals.extend(agent_goal.copy())
                starts.extend(agent_start.copy())
                state["agent_waypoints"].extend(agent_goal.copy())
                state["agent_observations"].extend(agent_start.copy())

            # Immutable objects - Should not change ever!
            state["composite_goals"] = goals.copy()
            state["composite_starts"] = starts.copy()
            logging.debug("Sampled the required starts and goals")

        all_done = False
        collided = [False for _ in range(num_agents)]
        agent_dones = [False for _ in range(num_agents)]

        while not all_done:

            state["goal"] = state["agent_waypoints"][0][1]
            state["grid"]["goal"] = state["agent_waypoints"][0][0]
            state["observation"] = state["agent_observations"][0][1]
            state["grid"]["observation"] = state["agent_observations"][0][0]

            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goals[0][0]
            eval_env.unwrapped.state_grid = state["grid"]["observation"]

            # NOTE: state's agent_observations, agent_waypoints and goal are updated
            if isinstance(policy, BasePolicy):
                actions, agent_goals = policy.select_action(state)
            plan_cost = cls._get_policy_cbs_plan_cost(policy)
            if plan_cost is not None:
                for record in augmented_ep_record_list:
                    if record.get("cbs_plan_cost") is None:
                        record["cbs_plan_cost"] = plan_cost
            plan_time = cls._get_policy_cbs_plan_time(policy)
            if plan_time is not None:
                for record in augmented_ep_record_list:
                    if record.get("cbs_plan_time") is None:
                        record["cbs_plan_time"] = plan_time

            wait_agents = []
            for agent_id in range(num_agents):

                if agent_dones[agent_id]:
                    continue

                if wait:
                    if agent_id in wait_agents:
                        continue

                    for other_agent_id in range(agent_id + 1, num_agents):
                        if agent_id == other_agent_id:
                            continue

                        # State here is agent ID x 2 where each element is a tuple of grid and visual observation
                        agent_state = np.array(state["agent_observations"][agent_id][0])
                        other_agent_state = np.array(
                            state["agent_observations"][other_agent_id][0]
                        )

                        if np.linalg.norm(agent_state - other_agent_state) < 1.0:
                            # Wait if the distance is less than MAX_ACTION
                            wait_agents.append(other_agent_id)

                if isinstance(policy, BasePolicy):
                    assert len(agent_goals[agent_id]) == 2 and isinstance(
                        agent_goals[agent_id], tuple
                    )
                    state["agent_waypoints"][agent_id] = agent_goals[agent_id]

                current_agent_waypoint = state["agent_waypoints"][agent_id]
                current_agent_observation = state["agent_observations"][agent_id]

                augmented_ep_waypoint_list[agent_id].append(current_agent_waypoint)
                augmented_ep_observation_list[agent_id].append(
                    current_agent_observation
                )

                state_copy = state.copy()

                state["goal"] = state_copy["agent_waypoints"][agent_id][1]
                state["grid"]["goal"] = state_copy["agent_waypoints"][agent_id][0]
                state["observation"] = state_copy["agent_observations"][agent_id][1]
                state["grid"]["observation"] = state_copy["agent_observations"][
                    agent_id
                ][0]

                if "goalconditioned" in type(eval_env.env).__name__.lower():
                    eval_env.env._goal = goals[agent_id][0]
                eval_env.unwrapped.state_grid = state["grid"]["observation"]

                action = (
                    actions[agent_id]
                    if isinstance(policy, BasePolicy)
                    else policy.select_action(state)
                )

                state, reward, done, info = eval_env.step(
                    np.copy(action), num_agents=num_agents
                )

                augmented_ep_record_list[agent_id]["steps"] += 1
                augmented_ep_record_list[agent_id]["rewards"] += reward

                agent_cost = info.get("cost", 0.0)
                if agent_cost > augmented_ep_record_list[agent_id]["max_step_cost"]:
                    augmented_ep_record_list[agent_id]["max_step_cost"] = agent_cost
                augmented_ep_record_list[agent_id]["cumulative_costs"] += agent_cost

                # At this point the state is changed and does not have the extra attributes so add them back
                state["composite_goals"] = state_copy["composite_goals"]
                state["composite_starts"] = state_copy["composite_starts"]

                state["agent_waypoints"] = state_copy["agent_waypoints"]
                state["agent_observations"] = state_copy["agent_observations"]

                # The agent's observations are updated based on the step function
                state["agent_observations"][agent_id] = (
                    state["grid"]["observation"],
                    state["observation"],
                )

                augmented_ep_reward_list[agent_id].append(reward)

                if done:
                    augmented_ep_record_list[agent_id]["success"] = info["success"]
                    terminal_agent_observation = (
                        info["terminal_observation"]["grid"]["observation"],
                        info["terminal_observation"]["observation"],
                    )
                    augmented_ep_observation_list[agent_id].append(
                        terminal_agent_observation
                    )
                    agent_dones[agent_id] = True

            # Check if any of the agent's positions are within some threshold
            for agent_id in range(num_agents):
                for other_agent_id in range(num_agents):
                    if (
                        agent_id == other_agent_id
                        or other_agent_id in wait_agents
                        or agent_dones[agent_id]
                        or agent_dones[other_agent_id]
                    ):
                        continue

                    agent_state = np.array(state["agent_observations"][agent_id][0])
                    other_agent_state = np.array(
                        state["agent_observations"][other_agent_id][0]
                    )

                    if np.linalg.norm(agent_state - other_agent_state) < threshold:
                        collided[agent_id] = True
                        collided[other_agent_id] = True
                        augmented_ep_record_list[agent_id]["success"] = False

            all_done = all(agent_dones)

        for agent_id in range(num_agents):
            if collided[agent_id]:
                logging.info(f"Agent {agent_id} was within threshold of another agent")
                augmented_ep_record_list[agent_id]["success"] = False
                break

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
            augmented_ep_record_list,
        )

    @classmethod
    def get_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        input_starts=None,
        input_goals=None,
        start_costs=None,
        threshold=0.05,
        habitat=False,
        wait=False,
    ):
        if habitat:
            if (
                input_starts is not None
                and input_goals is not None
                and start_costs is not None
            ):
                assert isinstance(input_starts, list) and isinstance(input_goals, list)
                assert len(input_starts[0]) == 2 and len(input_goals[0]) == 2

            return cls.get_visual_trajectories(
                policy,
                eval_env,
                num_agents,
                input_starts,
                input_goals,
                start_costs,
                threshold,
                wait=wait,
            )
        else:
            return cls.get_grid_trajectories(
                policy,
                eval_env,
                num_agents,
                input_starts,
                input_goals,
                start_costs,
                threshold,
                wait=wait,
            )
