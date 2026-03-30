import torch
import numpy as np

from pud.buffers.buffer import ReplayBuffer
from pud.algos.data_struct import inp_to_numpy


class VisualReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):
        super(VisualReplayBuffer, self).__init__(
            obs_dim=2,
            goal_dim=2,
            action_dim=action_dim,
            max_size=max_size,
        )

        obs_shape, goal_shape = None, None
        if isinstance(obs_dim, tuple):
            obs_shape = (max_size, *obs_dim)
            goal_shape = (max_size, *goal_dim)
        else:
            obs_shape = (max_size, obs_dim)
            goal_shape = (max_size, goal_dim)

        self.observation = np.zeros(obs_shape)
        self.goal = np.zeros(goal_shape)
        self.next_observation = np.zeros(obs_shape)
        self.next_goal = np.zeros(goal_shape)

        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        state = inp_to_numpy(state)
        next_state = inp_to_numpy(next_state)
        super().add(state, action, next_state, reward, done)


class ConstrainedVisualReplayBuffer(VisualReplayBuffer):
    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):
        super(ConstrainedVisualReplayBuffer, self).__init__(
            obs_dim, goal_dim, action_dim, max_size
        )
        self.cost = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, cost, done):
        self.cost[self.ptr] = cost
        super().add(state, action, next_state, reward, done)

    def sample_w_cost(self, batch_size):
        """A separate method to sample with cost, leave the original sample method for compat reason"""
        ind = np.random.randint(0, self.size, size=batch_size)

        batch = (
            dict(
                observation=torch.FloatTensor(self.observation[ind]),
                goal=torch.FloatTensor(self.goal[ind]),
            ),
            dict(
                observation=torch.FloatTensor(self.next_observation[ind]),
                goal=torch.FloatTensor(self.next_goal[ind]),
            ),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.cost[ind]),
            torch.FloatTensor(self.done[ind]),
        )
        return batch
