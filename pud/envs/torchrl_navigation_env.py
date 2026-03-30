import os
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt

from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs
from torchrl.envs import EnvBase, TransformedEnv, StepCounter
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)

from pud.envs.simple_navigation_env import WALLS, thin_walls, plot_walls, resize_walls


# Helper functions
def _normalize_obs(self, obs):
    return np.array([obs[:, 0] / float(self._height), obs[:, 1] / float(self._width)]).T


def _compute_apsp(self, walls):
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
    dist = np.full((height, width, height, width), np.float32("inf"))
    for (i1, j1), dist_dict in nx.shortest_path_length(g):
        for (i2, j2), d in dist_dict.items():
            dist[i1, j1, i2, j2] = d
    return dist


def _get_distance(self, obs, goal):
    discretized_obs = self._discretize_state(obs)
    discretized_goal = self._discretize_state(goal)
    return self._apsp[
        discretized_obs[:, 0],
        discretized_obs[:, 1],
        discretized_goal[:, 0],
        discretized_goal[:, 1],
    ]


def _discretize_state(self, state, resolution=1.0):
    discretized_states = np.floor(resolution * state).astype(np.int64)
    # Round down to the nearest cell if at the boundary.
    row_mask = discretized_states[:, 0] == self._height
    column_mask = discretized_states[:, 1] == self._width
    discretized_states[row_mask, 0] -= 1
    discretized_states[column_mask, 1] -= 1
    return discretized_states


def _discretize_single_state(self, state):
    (i, j) = np.floor(state).astype(np.int64)
    i = min(i, self._height - 1)
    j = min(j, self._width - 1)
    return (i, j)


def _is_blocked(self, agent_id, state):
    unbatched_obs_space = self.unbatched_observation_spec[
        "agents", "observation", "state"
    ][agent_id].space
    low = unbatched_obs_space.low.cpu().numpy()
    high = unbatched_obs_space.high.cpu().numpy()
    if not np.all(state >= low) or not np.all(state <= high):
        return True
    discretized_state = self._discretize_state(state)
    return np.all(self._walls[discretized_state[:, 0], discretized_state[:, 1]] == 1)


def _is_state_blocked(self, agent_id, state):
    obs_space = self.unbatched_observation_spec["agents", "observation", "state"][
        agent_id
    ].space
    low = obs_space.low.cpu().numpy()
    high = obs_space.high.cpu().numpy()
    if not np.all(state >= low) or not np.all(state <= high):
        return True
    (i, j) = self._discretize_single_state(state)
    return self._walls[i, j] == 1


def _sample_empty_state(self, agent_id):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states, size=self.batch_size[0])
    state = np.array(
        [candidate_states[0][state_index], candidate_states[1][state_index]],
        dtype=np.float32,
    ).T
    # state += np.random.uniform(size=2)
    assert not self._is_blocked(agent_id, state)
    return state


def _set_env_difficulty(self, difficulty):
    self._difficulty = difficulty
    self._set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=max(
            0, np.max(self._apsp[np.isfinite(self._apsp)]) * (self._difficulty - 0.05)
        ),
        max_dist=np.max(self._apsp[np.isfinite(self._apsp)])
        * (self._difficulty + 0.05),
    )


# Goal Sampling Functions
def _set_sample_goal_args(self, prob_constraint=None, min_dist=None, max_dist=None):
    assert min_dist is not None
    assert max_dist is not None
    assert min_dist >= 0
    assert max_dist >= min_dist
    assert prob_constraint is not None
    self._min_dist = min_dist
    self._max_dist = max_dist
    self._prob_constraint = prob_constraint


def _sample_goal(self, agent_id, obs):
    if np.random.random() < self._prob_constraint:
        return self._sample_goal_constrained(
            agent_id, obs, self._min_dist, self._max_dist
        )
    else:
        return self._sample_goal_unconstrained(agent_id, obs)


def _sample_goal_constrained(self, agent_id, obs, min_dist, max_dist):
    discretized_state = self._discretize_state(obs)
    mask = np.logical_and(
        self._apsp[discretized_state[:, 0], discretized_state[:, 1]] >= min_dist,
        self._apsp[discretized_state[:, 0], discretized_state[:, 1]] <= max_dist,
    )
    mask = np.logical_and(mask, self._walls == 0)
    goals = []
    for b in range(self.batch_size[0]):
        candidate_states = np.where(mask[b])
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (obs, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array(
            [candidate_states[0][goal_index], candidate_states[1][goal_index]],
            dtype=np.float32,
        ).T
        # goal += np.random.uniform(size=2)
        goals.append(goal)
    goals = np.array(goals)
    dist_to_goal = self._get_distance(obs, goals)
    if not (np.all(min_dist <= dist_to_goal) and np.all(dist_to_goal <= max_dist)):
        # Find the index of the goals that are not within the constraints
        bad_goal_index = np.where(
            np.logical_or(dist_to_goal < min_dist, dist_to_goal > max_dist)
        )[0][0]
        statement = f"Goals {goals[bad_goal_index]} are not within constraints."
        debug = (
            f"Dist: {dist_to_goal[bad_goal_index]}. Min: {min_dist}. Max: {max_dist}"
        )
        assert False, statement + debug
    if self._is_blocked(agent_id, goals):
        assert False, "Goals are blocked."
    return (obs, goals)


def _sample_goal_unconstrained(self, agent_id, obs):
    return (obs, self._sample_empty_state(agent_id))


# Mandatory environment functions
def _reset(self, tensordict):

    agent_states = []
    for agent_id in range(self.num_agents):
        count = 0
        agent_goal = None
        while agent_goal is None:
            agent_observation = self._sample_empty_state(agent_id)
            (agent_observation, agent_goal) = self._sample_goal(
                agent_id, agent_observation
            )
            count += 1
            if count > 1000:
                print("WARNING: Unable to find goal within constraints.")

        agent_observation = self._normalize_obs(agent_observation)
        agent_goal = self._normalize_obs(agent_goal)
        agent_tensordict = TensorDict(
            {
                "observation": TensorDict(
                    {
                        "state": torch.tensor(
                            agent_observation, dtype=torch.float32
                        ).to(self.device),
                        "goal": torch.tensor(agent_goal, dtype=torch.float32).to(
                            self.device
                        ),
                    },
                    batch_size=self.batch_size[0],
                ),
            },  # type: ignore
            batch_size=self.batch_size[0],
        )
        agent_states.append(agent_tensordict)
    out = TensorDict(
        {
            "agents": torch.stack(agent_states, dim=1).to(self.device),
        },
        batch_size=self.batch_size[0] if tensordict is None else tensordict.shape,
    )
    return out


def _step(self, tensordict):
    agent_dones = []
    agent_states = []

    num_substeps = 10
    dt = 1.0 / num_substeps
    batch_size = self.batch_size[0]
    height, width = self._height, self._width
    threshold_distance = self._threshold_distance

    for agent_id in range(self.num_agents):
        action = tensordict["agents", "action"][:, agent_id].squeeze(-1)
        goal = tensordict["agents", "observation", "goal"][:, agent_id]
        state = tensordict["agents", "observation", "state"][:, agent_id]

        if self.action_noise > 0:
            noise = torch.normal(
                0, self.action_noise, action.shape, device=action.device
            )
            action += noise

        action = torch.clamp(
            action,
            min=self.unbatched_action_spec["agents", "action"][agent_id].space.low.to(
                self.device
            ),
            max=self.unbatched_action_spec["agents", "action"][agent_id].space.high.to(
                self.device
            ),
        )

        goal_np = goal.cpu().numpy()
        state_np = state.cpu().numpy()
        denormalized_goal = goal_np * np.array([height, width])
        denormalized_state = state_np * np.array([height, width])

        for b in range(batch_size):
            for _ in np.linspace(0, 1, num_substeps):
                for axis in range(action.shape[-1]):
                    new_state = denormalized_state[b].copy()
                    new_state[axis] += dt * action[b, axis].cpu().numpy()
                    if not self._is_state_blocked(agent_id, new_state):
                        denormalized_state[b] = new_state

        agent_done = (
            np.linalg.norm(denormalized_goal - denormalized_state, axis=-1)
            < threshold_distance
        )
        agent_dones.append(agent_done)

        normalized_state = self._normalize_obs(denormalized_state)
        if self._reward_type == "dense":
            reward = -np.linalg.norm(goal_np - normalized_state, axis=-1).reshape(-1, 1)
        else:
            reward = -np.ones((batch_size, 1))

        td_state = TensorDict(
            {
                "observation": TensorDict(
                    {
                        "state": torch.tensor(normalized_state, dtype=torch.float32).to(
                            self.device
                        ),
                        "goal": torch.tensor(goal_np, dtype=torch.float32).to(
                            self.device
                        ),
                    },
                    batch_size=[batch_size],
                ),
                "reward": torch.tensor(reward, dtype=torch.float32).to(self.device),
            },  # type: ignore
            batch_size=[batch_size],
        )
        agent_states.append(td_state)

    agent_dones = np.array(agent_dones).astype(bool)
    out = TensorDict(
        {
            "agents": torch.stack(agent_states, dim=1).to(self.device),
            "done": torch.all(torch.tensor(agent_dones).T, dim=-1).to(self.device),
        },
        batch_size=tensordict.shape if tensordict is not None else [batch_size],
    )
    return out


def _make_spec(self):

    self.agent_observation_spec = CompositeSpec(
        state=BoundedTensorSpec(
            low=torch.tensor([0, 0], dtype=torch.float32),
            high=torch.tensor([self._height, self._width], dtype=torch.float32),
        ),
        goal=BoundedTensorSpec(
            low=torch.tensor([0, 0], dtype=torch.float32),
            high=torch.tensor([self._height, self._width], dtype=torch.float32),
        ),
    )
    self.agent_action_spec = BoundedTensorSpec(
        low=torch.tensor([-1.0, -1.0], dtype=torch.float32),
        high=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    self.agent_reward_spec = UnboundedContinuousTensorSpec(
        shape=torch.Size((1,)), dtype=torch.float32
    )

    action_specs = []
    reward_specs = []
    observation_specs = []
    for agent_id in range(self.num_agents):
        action_specs.append(self.agent_action_spec)
        reward_specs.append(self.agent_reward_spec)
        observation_specs.append(self.agent_observation_spec)

    self.unbatched_action_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {"action": torch.stack(action_specs, dim=0)},
                shape=torch.Size([self.num_agents]),
            )
        },
    )
    self.unbatched_reward_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {"reward": torch.stack(reward_specs, dim=0)},
                shape=torch.Size([self.num_agents]),
            )
        },
    )
    self.unbatched_observation_spec = CompositeSpec(
        {
            "agents": CompositeSpec(
                {"observation": torch.stack(observation_specs, dim=0)},
                shape=torch.Size([self.num_agents]),
            )
        },
    )
    self.unbatched_done_spec = DiscreteTensorSpec(
        n=2, shape=torch.Size((1,)), dtype=torch.bool
    )

    self.action_spec = self.unbatched_action_spec.expand(
        *self.batch_size, *self.unbatched_action_spec.shape
    )
    self.reward_spec = self.unbatched_reward_spec.expand(
        *self.batch_size, *self.unbatched_reward_spec.shape
    )
    self.observation_spec = self.unbatched_observation_spec.expand(
        *self.batch_size, *self.unbatched_observation_spec.shape
    )
    self.done_spec = self.unbatched_done_spec.expand(
        *self.batch_size, *self.unbatched_done_spec.shape
    )

    self.group_map = {"agents": [str(i) for i in range(self.num_agents)]}


def _set_seed(self, seed):
    rng = torch.manual_seed(seed)
    self.rng = rng


def _render(self, tensordict, mode="human"):
    if mode == "human":
        fig, ax = plt.subplots()
        ax = plot_walls(self._walls, ax)
        agent_colors = ["b", "r", "g", "c", "m", "y", "k"]
        for agent_id in range(self.num_agents):
            agent_state = (
                tensordict["agents", "observation", "state"][agent_id].cpu().numpy()
            )
            agent_goal = (
                tensordict["agents", "observation", "goal"][agent_id].cpu().numpy()
            )
            plt.plot(
                agent_state[0], agent_state[1], color=agent_colors[agent_id], marker="o"
            )
            plt.plot(
                agent_goal[0], agent_goal[1], color=agent_colors[agent_id], marker="x"
            )
            plt.show(block=False)
    elif mode == "rgb_array":
        fig, ax = plt.subplots()
        ax = plot_walls(self._walls, ax)
        agent_colors = ["b", "r", "g", "c", "m", "y", "k"]
        for agent_id in range(self.num_agents):
            agent_state = (
                tensordict["agents", "observation", "state"][0][agent_id].cpu().numpy()
            )
            agent_goal = (
                tensordict["agents", "observation", "goal"][0][agent_id].cpu().numpy()
            )
            plt.plot(
                agent_state[0], agent_state[1], color=agent_colors[agent_id], marker="o"
            )
            plt.plot(
                agent_goal[0], agent_goal[1], color=agent_colors[agent_id], marker="x"
            )
        fig = plt.gcf()
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")


class MultiAgentPointEnv(EnvBase):

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        walls,
        num_agents=2,
        batch_size=10,
        seed=None,
        resize_factor=5,
        thin=False,
        action_noise=1.0,
        apsp_path=None,
        reward_type="dense",
        device="cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([batch_size]))

        if thin and resize_factor > 1:
            self._walls = thin_walls(WALLS[walls], resize_factor)
        elif not thin and resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]

        self.num_agents = num_agents
        self.action_noise = action_noise

        self._min_dist = 0
        self._max_dist = 4
        self._difficulty = 0.5
        self._prob_constraint = 0.8
        self._threshold_distance = 1.0
        self._reward_type = reward_type
        self._apsp_path = apsp_path

        print("Computing all-pairs shortest paths.")
        if self._apsp_path is None or self._apsp_path == "":
            apsp_pickle_str = (
                "pud/envs/precompiles/" + walls + "_" + str(resize_factor) + "_apsp.pkl"
            )
            self._apsp_path = Path(os.getcwd()).parent.parent.parent / apsp_pickle_str
            self._apsp = self._compute_apsp(self._walls)
            import pickle

            with open(self._apsp_path, "wb") as f:
                pickle.dump(self._apsp, f)
        else:
            self._apsp_path = Path(os.getcwd()).parent.parent.parent / self._apsp_path
            import pickle

            with open(self._apsp_path, "rb") as f:
                self._apsp = pickle.load(f)
        print("Done computing all-pairs shortest paths.")

        (self._height, self._width) = self._walls.shape

        self._set_sample_goal_args(
            prob_constraint=self._prob_constraint,
            min_dist=max(0, self.max_goal_dist * (self._difficulty - 0.05)),
            max_dist=self.max_goal_dist * (self._difficulty + 0.05),
        )

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)  # type: ignore

    _step = _step
    _set_seed = _set_seed
    _make_spec = _make_spec
    _reset = _reset  # type: ignore
    _render = _render

    _get_distance = _get_distance
    _is_blocked = _is_blocked
    _discretize_state = _discretize_state
    _compute_apsp = _compute_apsp
    _normalize_obs = _normalize_obs
    _set_env_difficulty = _set_env_difficulty
    _is_state_blocked = _is_state_blocked
    _sample_empty_state = _sample_empty_state
    _discretize_single_state = _discretize_single_state

    _sample_goal = _sample_goal
    _set_sample_goal_args = _set_sample_goal_args
    _sample_goal_constrained = _sample_goal_constrained
    _sample_goal_unconstrained = _sample_goal_unconstrained

    @property
    def max_goal_dist(self):
        return np.max(self._apsp[np.isfinite(self._apsp)])


if __name__ == "__main__":

    env = MultiAgentPointEnv(
        walls="CentralObstacle",
        num_agents=2,
        seed=0,
        resize_factor=5,
        thin=False,
        action_noise=1.0,
        device="cuda:0",
    )
    check_env_specs(env)

    transformed_env = TransformedEnv(env, StepCounter(max_steps=20))
    check_env_specs(transformed_env)
