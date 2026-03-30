import logging
import numpy as np

from pud.algos.policies import BasePolicy


class Collector:
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        self.buffer = buffer
        self.env = env
        self.policy = policy

        self.steps = 0
        self.state = env.reset()
        self.initial_collect_steps = initial_collect_steps

    def step(self, num_steps):
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)

            next_state, reward, done, info = self.env.step(action)
            if info.get("last_step", False):
                self.buffer.add(
                    self.state, action, info["terminal_observation"], reward, done
                )
                self.state = next_state
            else:
                self.buffer.add(self.state, action, next_state, reward, done)
                self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            rb_vec.append(eval_env.reset())
        rb_vec = np.array([x["observation"] for x in rb_vec])
        return rb_vec

    @classmethod
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        c = 0
        r = 0
        rewards = []
        state = eval_env.reset()
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode:
                c += 1

            r += reward
            if done:
                rewards.append(r)
                if by_episode:
                    c += 1
                r = 0
        return rewards

    @classmethod
    def step_cleanup(cls, search_policy, eval_env, num_steps):
        c = 0
        while c < num_steps:
            goal = search_policy.get_goal_in_rb()
            state = eval_env.reset()
            done = False

            while True:
                state["goal"] = goal
                try:
                    action = search_policy.select_action(state)
                except Exception as e:
                    raise e

                state, _, done, _ = eval_env.step(np.copy(action))
                c += 1

                if done or c >= num_steps or search_policy.reached_final_waypoint:
                    break

    @classmethod
    def get_grid_trajectory(cls, policy, eval_env, start=None, goal=None):
        ep_reward_list = []
        ep_waypoint_list = []
        ep_observation_list = []

        state = eval_env.reset()

        denormalize_factor = np.array(
            [eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32
        )

        if start is not None and goal is not None:
            state["goal"] = goal.copy()
            state["observation"] = start.copy()
            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goal * denormalize_factor
            eval_env.unwrapped.state = state["observation"] * denormalize_factor

        ep_goal = state["goal"]
        ep_start = state["observation"]
        ep_record = {"rewards": 0.0, "steps": 0}

        while True:
            ep_observation_list.append(state["observation"])
            action = policy.select_action(state)  # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state["goal"])
            state, reward, done, info = eval_env.step(np.copy(action))
            ep_record["steps"] += 1
            ep_record["rewards"] += reward
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
    def get_visual_trajectory(cls, policy, eval_env, start=None, goal=None):
        ep_reward_list = []
        ep_waypoint_list = []
        ep_observation_list = []

        state, info = eval_env.reset()

        if start is not None and goal is not None:
            state["goal"] = goal[1].copy()
            state["grid"]["goal"] = goal[0].copy()
            state["observation"] = start[1].copy()
            state["grid"]["observation"] = start[0].copy()
            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goal[0]
            eval_env.unwrapped.state_grid = state["grid"]["observation"]

        ep_record = {"rewards": 0.0, "steps": 0}
        ep_goal = (state["grid"]["goal"], state["goal"])
        ep_start = (state["grid"]["observation"], state["observation"])
        while True:
            ep_observation = (state["grid"]["observation"], state["observation"])
            ep_observation_list.append(ep_observation)

            action = policy.select_action(state)  # NOTE: state['goal'] may be modified

            ep_waypoint = (state["grid"]["goal"], state["goal"])
            ep_waypoint_list.append(ep_waypoint)

            state, reward, done, info = eval_env.step(np.copy(action))
            ep_record["steps"] += 1
            ep_record["rewards"] += reward
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
        assert start_cost is None
        if habitat:
            if input_start is not None and input_goal is not None:
                assert len(input_start) == 2 and len(input_goal[0]) == 2
            return cls.get_visual_trajectory(policy, eval_env, input_start, input_goal)
        else:
            return cls.get_grid_trajectory(policy, eval_env, input_start, input_goal)

    @classmethod
    def get_grid_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        starts=None,
        goals=None,
        threshold: float = 0.05,
        wait=False
    ):

        augmented_ep_reward_list = [[] for _ in range(num_agents)]
        augmented_ep_waypoint_list = [[] for _ in range(num_agents)]
        augmented_ep_observation_list = [[] for _ in range(num_agents)]
        augmented_ep_records_list = [
            {"rewards": 0.0, "steps": 0} for _ in range(num_agents)
        ]

        denormalize_factor = np.array(
            [eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32
        )

        state = eval_env.reset()

        if starts is not None and goals is not None:

            state["goal"] = goals.copy()
            state["observation"] = starts.copy()

            state["composite_goals"] = goals.copy()
            state["agent_waypoints"] = goals.copy()

            state["composite_starts"] = starts.copy()
            state["agent_observations"] = starts.copy()

        else:
            # Use the sampled start and goal for the first agent
            agent_goal = [state["goal"]]
            agent_start = [state["observation"]]

            # Mutable objects
            state["agent_waypoints"] = agent_goal.copy()
            state["agent_observations"] = agent_start.copy()

            goals = agent_goal.copy()
            starts = agent_start.copy()

            # Sample the starts and goals for the other agents
            for _ in range(num_agents - 1):

                agent_state = eval_env.reset()
                agent_goal = [agent_state["goal"]]
                agent_start = [agent_state["observation"]]

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

            state["goal"] = state["agent_waypoints"][0]
            state["observation"] = state["agent_observations"][0]

            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goals[0] * denormalize_factor
            eval_env.unwrapped.state = state["observation"] * denormalize_factor

            # NOTE: state's agent_observations, agent_waypoints and goal are updated
            if isinstance(policy, BasePolicy):
                actions, agent_goals = policy.select_action(state)

            wait_agents = []
            for agent_id in range(num_agents):

                if agent_dones[agent_id]:
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

                augmented_ep_records_list[agent_id]["steps"] += 1
                augmented_ep_records_list[agent_id]["rewards"] += reward

                # At this point the state is changed and does not have the extra attributes so add them back
                state["composite_goals"] = state_copy["composite_goals"]
                state["composite_starts"] = state_copy["composite_starts"]

                state["agent_waypoints"] = state_copy["agent_waypoints"]
                state["agent_observations"] = state_copy["agent_observations"]

                # The agent's observations are updated based on the step function
                state["agent_observations"][agent_id] = state["observation"]

                augmented_ep_reward_list[agent_id].append(reward)

                if done:
                    augmented_ep_records_list[agent_id]["success"] = info["success"]
                    terminal_agent_observation = info["terminal_observation"][
                        "observation"
                    ]
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

                    agent_state = np.array(state["agent_observations"][agent_id])
                    other_agent_state = np.array(
                        state["agent_observations"][other_agent_id]
                    )

                    if np.linalg.norm(agent_state - other_agent_state) < threshold:
                        collided[agent_id] = True
                        collided[other_agent_id] = True
                        augmented_ep_records_list[agent_id]["success"] = False

            all_done = all(agent_dones)

        for agent_id in range(num_agents):
            if collided[agent_id]:
                logging.info(f"Agent {agent_id} was within threshold of another agent")
                augmented_ep_records_list[agent_id]["success"] = False
                break

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
            augmented_ep_records_list,
        )

    @classmethod
    def get_visual_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        starts=None,
        goals=None,
        threshold: float = 0.05,
        wait=False,
    ):

        augmented_ep_reward_list = [[] for _ in range(num_agents)]
        augmented_ep_waypoint_list = [[] for _ in range(num_agents)]
        augmented_ep_observation_list = [[] for _ in range(num_agents)]
        augmented_ep_records_list = [
            {"rewards": 0.0, "steps": 0} for _ in range(num_agents)
        ]

        state, info = eval_env.reset()

        if starts is not None and goals is not None:

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
        else:
            # Use the sampled start and goal for the first agent
            agent_goal = [(state["grid"]["goal"], state["goal"])]
            agent_start = [(state["grid"]["observation"], state["observation"])]

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

            state["goal"] = state["agent_waypoints"][1]
            state["grid"]["goal"] = state["agent_waypoints"][0]
            state["observation"] = state["agent_observations"][1]
            state["grid"]["observation"] = state["agent_observations"][0]

            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goals[0][0]
            eval_env.unwrapped.state_grid = state["grid"]["observation"]

            # NOTE: state's agent_observations, agent_waypoints and goal are updated
            if isinstance(policy, BasePolicy):
                actions, agent_goals = policy.select_action(state)

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

                augmented_ep_records_list[agent_id]["steps"] += 1
                augmented_ep_records_list[agent_id]["rewards"] += reward

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
                    augmented_ep_records_list[agent_id]["success"] = info["success"]
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
                        augmented_ep_records_list[agent_id]["success"] = False

            all_done = all(agent_dones)

        for agent_id in range(num_agents):
            if collided[agent_id]:
                logging.info(f"Agent {agent_id} was within threhsold of another agent")
                augmented_ep_records_list[agent_id]["success"] = False
                break

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
            augmented_ep_records_list,
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
        wait=False
    ):
        assert start_costs is None
        if habitat:
            if input_starts is not None and input_goals is not None:
                assert isinstance(input_starts, list) and isinstance(input_goals, list)
                assert len(input_starts[0]) == 2 and len(input_goals[0]) == 2

            return cls.get_visual_trajectories(
                policy,
                eval_env,
                num_agents,
                starts=input_starts,
                goals=input_goals,
                threshold=threshold,
                wait=wait,
            )
        else:
            return cls.get_grid_trajectories(
                policy,
                eval_env,
                num_agents,
                starts=input_starts,
                goals=input_goals,
                threshold=threshold,
                wait=wait,
            )
