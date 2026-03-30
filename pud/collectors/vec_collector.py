import gym
import numpy as np
from typing import List
from copy import deepcopy


class VectorCollector:
    def __init__(self, policy, buffer, envs: List[gym.Env], initial_collect_steps=0):
        self.buffer = buffer
        self.env = envs
        self.num_envs = len(envs)
        self.policy = policy

        self.steps = 0
        self.states = [env.reset() for env in envs]
        self.initial_collect_steps = initial_collect_steps

        assert (
            self.initial_collect_steps % self.num_envs == 0
        ), "initial collect steps must be divisible by num_envs"

    def step(self, num_steps):
        num_steps = num_steps // self.num_envs
        actions = None
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps // self.num_envs:
                actions = [env.action_space.sample() for env in self.env]
            else:
                b_states = {
                    "observation": np.stack(
                        [st["observation"] for st in self.states], axis=0  # type: ignore
                    ),
                    "goal": np.stack([st["goal"] for st in self.states], axis=0),  # type: ignore
                }
                actions = self.policy.select_action(b_states)
                if len(actions.shape) == 1:
                    actions = np.reshape(actions, [1, -1])

            next_states = [None] * self.num_envs
            for ii, act in enumerate(actions):
                env_ii = self.env[ii]
                result = env_ii.step(np.copy(act))
                if len(result) == 5:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                else:
                    next_state, reward, done, info = result
                next_states[ii] = next_state
                if info.get("last_step", False):
                    self.buffer.add(
                        self.states[ii], act, info["terminal_observation"], reward, done
                    )
                else:
                    self.buffer.add(self.states[ii], act, next_state, reward, done)

                self.steps += 1

            self.states = next_states

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
            state, reward, done, _ = eval_env.step(np.copy(action))
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
    def eval_agent_n_trajs(cls, policy, eval_env, n, by_episode=True, verbose=False):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        c = 0
        r = 0
        traj = []

        rewards = []
        trajs = []
        success = []

        state = eval_env.reset()
        traj.append(state)
        while c < n:
            action = policy.select_action(state)
            if verbose:
                print("episode {}, action: {}".format(c, action))

            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode:
                c += 1

            if not done:
                traj.append(deepcopy(state))
            else:
                traj.append(info["terminal_observation"])

            if verbose:
                print(
                    "obs:{}, action:{} goal:{}".format(
                        info["grid"]["observation"], action, info["grid"]["goal"]
                    )
                )

            r += reward
            if done:
                rewards.append(r)
                if by_episode:
                    c += 1
                r = 0
                trajs.append(traj)
                success.append(not info["timed_out"])
                traj = []
                if verbose:
                    print("#" * 15)
        return {"rewards": rewards, "trajs": trajs, "success": success}

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
    def get_trajectory(cls, policy, eval_env):
        ep_observation_list = []
        ep_waypoint_list = []
        ep_reward_list = []

        state = eval_env.reset()
        ep_goal = state["goal"]
        while True:
            ep_observation_list.append(state["observation"])
            action = policy.select_action(state)  # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state["goal"])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info["terminal_observation"]["observation"])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list
