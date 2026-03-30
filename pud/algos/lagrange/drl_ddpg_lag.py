import copy
import torch
import functools
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Union

from pud.algos.ddpg import UVFDDPG
from pud.algos.lagrange.lagrange import Lagrange
from pud.algos.data_struct import inp_to_torch_device
from pud.algos.distributional_ops import CategoricalActivation
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.algos.ddpg import EnsembledCritic, GoalConditionedCritic

nn = torch.nn
F = nn.functional


class DRLDDPGLag(UVFDDPG):
    """
    Base Distributional DDPG, removing unnecessary compat to non-distributional RL code
    Aim to have a clean separation from unconstrained and constrained, have costs in separate loops,
    even at the cost of efficiency
    """

    def __init__(
        self,
        # DDPG args
        state_dim,
        action_dim,
        max_action,
        discount=1,
        actor_update_interval=1,
        targets_update_interval=1,
        tau=0.005,
        CriticCls=GoalConditionedCritic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        # UVFDDPG args
        num_bins=20,
        use_distributional_rl=True,
        ensemble_size=3,
        # Cost configs
        cost_min: float = 0,
        cost_max: float = 2.0,
        cost_N=20,
        cost_critic_lr: float = 1e-3,
        cost_limit: float = 1.0,
        lambda_lr: float = 0.001,
        lambda_optimizer: str = "Adam",
        device: torch.device = torch.device("cpu"),
    ):
        super(DRLDDPGLag, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            actor_update_interval=actor_update_interval,
            targets_update_interval=targets_update_interval,
            tau=tau,
            num_bins=num_bins,
            use_distributional_rl=use_distributional_rl,
            ensemble_size=ensemble_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )

        self.lagrange = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=1e-3,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
        )
        self.lagrange_on = False
        self.device = device
        self.constraints = dict(min_cost=cost_min, max_cost=cost_max)

        # Add cost critic
        CostCriticCls = functools.partial(CriticCls, output_dim=cost_N)
        self.F_categorical = CategoricalActivation(
            vmin=cost_min, vmax=cost_max, N=cost_N
        )
        self.cost_critic = CostCriticCls(state_dim, action_dim)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=cost_critic_lr, eps=1e-07
        )

        if self.ensemble_size > 1:
            self.cost_critic = EnsembledCritic(
                self.cost_critic, ensemble_size=ensemble_size
            )
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            for i in range(
                1, len(self.cost_critic.critics)
            ):  # First copy already added
                cost_critic_copy = self.cost_critic.critics[i]
                self.cost_critic_optimizer.add_param_group(
                    {"params": cost_critic_copy.parameters()}
                )

    def select_action(self, state):
        """enable options on cuda, move input to self.device"""
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state["observation"].reshape(1, -1)),
                goal=torch.FloatTensor(state["goal"].reshape(1, -1)),
            )
            state = inp_to_torch_device(state, self.device)
            return self.actor(state).cpu().detach().numpy().flatten()

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state["observation"]),
                goal=torch.FloatTensor(state["goal"]),
            )
            state = inp_to_torch_device(state, self.device)
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)

    def update_cost_critic_target(self):
        for param, target_param in zip(
            self.cost_critic.parameters(), self.cost_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def set_lag_status(self, turn_on_lag: bool):
        self.lagrange_on = turn_on_lag

    def optimize(
        self,
        replay_buffer: ConstrainedReplayBuffer,
        iterations=1,
        batch_size=128,
    ):
        """
        Override the default optimize to include the Lagrange optimize and have an unified
        self.optimize_iterations count
        """
        opt_info = dict(actor_loss=[], critic_loss=[], cost_critic_loss=[])
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches
            state, next_state, action, reward, cost, done = replay_buffer.sample_w_cost(
                batch_size
            )
            state = inp_to_torch_device(state, self.device)
            next_state = inp_to_torch_device(next_state, self.device)
            action = inp_to_torch_device(action, self.device)
            reward = inp_to_torch_device(reward, self.device)
            cost = inp_to_torch_device(cost, self.device)
            done = inp_to_torch_device(done, self.device)

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info["critic_loss"].append(critic_loss.cpu().detach().numpy())

            # For cost critic
            cost_current_q = self.cost_critic(state, action)
            cost_target_q = self.cost_critic_target(
                next_state, self.actor_target(next_state)
            )
            cost_critic_loss = self.cost_critic_loss(
                cost_current_q, cost_target_q, torch.Tensor(cost), torch.Tensor(done)
            )
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()
            opt_info["cost_critic_loss"].append(cost_critic_loss.cpu().detach().numpy())

            if self.optimize_iterations % self.actor_update_interval == 0:
                # Compute actor loss
                actor_loss_r = -self.get_q_values(state)
                actor_loss = 0.0  # placeholder
                if self.lagrange_on:
                    actor_loss_c = self.get_cost_q_values(state)
                    mask_loss_c = actor_loss_c > self.lagrange.cost_limit
                    lag = self.lagrange.lagrangian_multiplier.item()
                    masked_actor_loss_c = (actor_loss_c * mask_loss_c).mean() * lag
                    actor_loss = (actor_loss_r.mean() + masked_actor_loss_c) / (1 + lag)
                else:
                    actor_loss = actor_loss_r.mean()  # Debug training, drop lagrange
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info["actor_loss"].append(actor_loss.cpu().detach().numpy())

            # Update the frozen target models
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()
                self.update_cost_critic_target()

        return opt_info

    def optimize_lagrange(self, ep_cost: float):
        """
        Wrapper of lagrange update, lagrange is updated separately as there are different options:
        1. update lagrange when new episode finishes
        2. update lagrange along with the main optimize call regardless new episode has finished
        NOTE: The omnisafe ddpg updates lagrange multiplier regardless a new episode has finished
        """
        self.lagrange.update_lagrange_multiplier(ep_cost)
        return self.lagrange.lagrangian_multiplier.item()

    def cost_critic_loss(
        self,
        current_q: Union[torch.Tensor, List[torch.Tensor]],
        target_q: Union[torch.Tensor, List[torch.Tensor]],
        cost: Union[torch.Tensor, torch.FloatTensor],  # (N, 1)
        done: torch.Tensor,  # (N, 1)
    ):
        """
        Loss on cumulative costs
        current_q: a torch tensor if the cost critic is not an ensemble, or a list of torch tensors if the cost critic
        is an ensemble that contains the outputs of all the cost critic, (N, 1)

        """
        current_q_list = current_q
        target_q_list = target_q
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            # Compute distributional td targets
            new_target_probs = None
            with torch.no_grad():
                target_q_probs = F.softmax(target_q, dim=1)  # type: ignore
                batch_size = target_q_probs.shape[0]
                zs = self.F_categorical.zs.tile([batch_size, 1]).to(self.device)
                new_zs = cost + (
                    (1 - done) * self.discount * zs
                )  # batch_size, num_classes
                new_target_probs = self.F_categorical.forward(
                    probs=target_q_probs, new_zs=new_zs
                )
            # cross entry loss: $$-\sum_{i}m_{i}\log p_{i}\left(x_{t},a_{t}\right)$$
            critic_loss = torch.mean(
                -torch.sum(
                    new_target_probs * torch.log_softmax(current_q, dim=1), dim=1
                )
            )
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def _get_cost_q_values(self, state):
        actions = self.actor(state)
        q_values = self.cost_critic(state, actions)
        return q_values

    def get_cost_q_values(
        self,
        state: Union[
            NDArray,
            torch.FloatTensor,
            Dict[str, NDArray],
            Dict[str, torch.Tensor],
            Dict[str, torch.FloatTensor],
        ],
        aggregate="mean",
    ):
        q_values = self._get_cost_q_values(state)
        q_values_list = []
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        expected_q_values_list = []
        for q_values in q_values_list:
            q_probs = F.softmax(q_values, dim=1)
            batch_size = q_probs.shape[0]
            zs = self.F_categorical.zs.tile([batch_size, 1]).to(self.device)
            # Take the inner product between these two tensors
            expected_q_values = torch.sum(q_probs * zs, dim=1, keepdim=True)
            expected_q_values_list.append(expected_q_values)

        expected_q_values = torch.stack(expected_q_values_list)
        if aggregate is not None:
            if aggregate == "mean":
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == "min":
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            elif aggregate == "max":
                expected_q_values, _ = torch.max(expected_q_values, dim=0)
            else:
                raise ValueError
        return expected_q_values

    def get_cost_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state["observation"]),
                goal=torch.FloatTensor(state["goal"]),
            )
            state = inp_to_torch_device(state, self.device)
            q_values = self.get_cost_q_values(state, **kwargs)
            return q_values.cpu().detach().numpy().squeeze(-1)

    def get_pairwise_cost(
        self, obs_vec, goal_vec=None, aggregate: Union[str, None] = "mean"
    ):
        """
        Estimates the pairwise costs. Return ensemble_size, obs_vec, goal_vec

        obs_vec: Array containing observations
        goal_vec: (optional) Array containing a second set of observations. If
                  not specified, computes the pairwise distances between obs_tensor and
                  itself.
        aggregate: (str) How to combine the predictions from the ensemble. Options
                   are to take the minimum predicted q value (i.e., the maximum distance),
                   the mean, or to simply return all the predictions.
        """
        if goal_vec is None:
            goal_vec = obs_vec

        dist_matrix = []
        for obs_index in range(len(obs_vec)):
            obs = obs_vec[obs_index]
            obs_repeat_tensor = np.repeat([obs], len(goal_vec), axis=0)
            state = {"observation": obs_repeat_tensor, "goal": goal_vec}
            dist = self.get_cost_to_goal(state, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = np.stack(dist_matrix)
        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])

        return pairwise_dist

    def state_dict(self):
        out = super().state_dict()
        out["cost_critic"] = self.cost_critic.state_dict()
        out["cost_critic_optimizer"] = self.cost_critic_optimizer.state_dict()
        return out

    def load_state_dict(self, state_dict: dict):
        unconstrained_keys = [
            "actor",
            "actor_optimizer",
            "critic",
            "critic_optimizer",
            "optimize_iterations",
        ]
        unconstrained_state_dict = {}
        for key in unconstrained_keys:
            unconstrained_state_dict[key] = state_dict[key]
        super().load_state_dict(unconstrained_state_dict)

        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic_optimizer.load_state_dict(state_dict["cost_critic_optimizer"])
