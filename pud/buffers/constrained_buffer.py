import torch
import numpy as np

from pud.buffers.buffer import ReplayBuffer


class ConstrainedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):
        """Maintain the
            - cost buffer
            - cost class indices
                - if distributional RL is used, which is a necessity to obtain good goal-conditioned cost critic

        Args:
            obs_dim (_type_): _description_
            goal_dim (_type_): _description_
            action_dim (_type_): _description_
            max_size (_type_, optional): _description_. Defaults to int(1e6).
        """
        super(ConstrainedReplayBuffer, self).__init__(
            obs_dim, goal_dim, action_dim, max_size
        )
        self.cost = np.zeros((max_size, 1))

    def add(
        self,
        state: dict,
        action: np.ndarray,
        next_state: dict,
        reward: float,
        cost: float,
        done: bool,
    ):
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
