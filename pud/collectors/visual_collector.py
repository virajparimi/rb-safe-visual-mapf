import numpy as np
from typing import Union
from copy import deepcopy

from pud.collectors.collector import Collector


def eval_agent_from_Q(policy, eval_env, collect_trajs=False):
    """
    Run evaluation and records the initial states for each episode
    until the pb Q from the env is empty
    """
    # Verify the eval_env has an non-empty Q of pbs
    assert hasattr(eval_env, "pb_Q")
    """At the end of the last pb in the Q, the step will trigger another
    reset, and it should be safely handled by the reset_orig, suppress the warning
    message by turning off verbose"""
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
            records[key]["traj"] = [init_state["grid"]["observation"]]
        records[key]["init_states"] = init_state["grid"]["observation"]
        return key

    c = 0  # count
    n = eval_env.get_Q_size()

    if n == 0:
        return records

    state, info = eval_env.reset()
    cur_key = new_record(state, info)

    while c < n:
        action = policy.select_action(state)
        """when episode ends:
        - state is the new state of the new epsiode
        - reward, done, info are from the last step of the terminated epsiode
        """
        state, reward, done, info = eval_env.step(action)

        records[cur_key]["steps"] += 1
        records[cur_key]["rewards"] += reward

        if (not done) and collect_trajs:
            records[cur_key]["traj"].append(state["grid"]["observation"])

        co = info.get("cost", 0.0)
        if co > records[cur_key]["max_step_cost"]:
            records[cur_key]["max_step_cost"] = co
        records[cur_key]["cum_costs"] += co

        if done:
            records[cur_key]["success"] = info["success"]
            if collect_trajs and "terminal_observation" in info:
                records[cur_key]["traj"].append(
                    info["terminal_observation"]["grid"]["observation"]
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


class VisualCollector(Collector):
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        super(VisualCollector, self).__init__(
            policy, buffer, env, initial_collect_steps=initial_collect_steps
        )

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


class ConstrainedVisualCollector(VisualCollector):
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        super(ConstrainedVisualCollector, self).__init__(
            policy, buffer, env, initial_collect_steps=initial_collect_steps
        )

        self.past_eps = []
        self.num_eps = 0
        self.state, info = env.reset()
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
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """
        c = 0  # count
        r = 0  # reward
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

        c = 0  # count

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
