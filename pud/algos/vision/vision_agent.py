import copy
import torch
import functools
import numpy as np
from torch import nn
from numpy.typing import NDArray
from torch.nn import functional as F
from typing import Literal, Optional, Union, Dict, List

from pud.utils import variance_initializer_
from pud.algos.lagrange.lagrange import Lagrange
from pud.algos.data_struct import inp_to_torch_device
from pud.algos.ddpg import merge_obs_goal, EnsembledCritic
from pud.algos.distributional_ops import CategoricalActivation
from pud.algos.vision.visual_models import VisualRGBEncoder, VisualEncoder
from pud.buffers.visual_buffer import VisualReplayBuffer, ConstrainedVisualReplayBuffer


class VisualActor(nn.Module):  # TODO: [256, 256], MLP class
    def __init__(
        self,
        state_dim: int,
        width: int,
        height: int,
        action_dim,
        max_action,
        embedding_size: int = 256,
        act_fn=nn.SELU,
        in_channels: int = 4,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        device=torch.device("cpu"),
    ):
        super().__init__()
        if encoder == "VisualEncoder":
            self.encoder = VisualEncoder(
                in_channels=in_channels,
                embedding_size=embedding_size,
                width=width,
                height=height,
                act_fn=act_fn,
                device=torch.device(device) if isinstance(device, str) else device,
            )
        elif encoder == "VisualRGBEncoder":
            self.encoder = VisualRGBEncoder(
                in_channels=in_channels,
                embedding_size=embedding_size,
                width=width,
                height=height,
                act_fn=act_fn,
                device=torch.device(device) if isinstance(device, str) else device,
            )
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.device = device
        self.max_action = max_action
        self.reset_parameters()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = self.max_action * a
        return a

    def reset_parameters(self):
        self.encoder.reset_parameters()
        variance_initializer_(
            self.l1.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(
            self.l2.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)


class VisualCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        width: int,
        height: int,
        action_dim,
        embedding_size: int,
        act_fn=nn.SELU,
        output_dim=1,
        in_channels: int = 4,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        device=torch.device("cpu"),
    ):
        super().__init__()

        if encoder == "VisualEncoder":
            self.encoder = VisualEncoder(
                in_channels=in_channels,
                embedding_size=embedding_size,
                width=width,
                height=height,
                act_fn=act_fn,
                device=device,
            )
        elif encoder == "VisualRGBEncoder":
            self.encoder = VisualRGBEncoder(
                in_channels=in_channels,
                embedding_size=embedding_size,
                width=width,
                height=height,
                act_fn=act_fn,
                device=device,
            )
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256 + action_dim, 256)
        self.l3 = nn.Linear(256, output_dim)
        self.device = device

        self.reset_parameters()

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], dim=1)))
        q = self.l3(q)
        return q

    def reset_parameters(self):
        self.encoder.reset_parameters()
        variance_initializer_(
            self.l1.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(
            self.l2.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)


class VisualGoalConditionedActor(VisualActor):
    def forward(self, state):
        latent_state = self.encoder.get_latent_state(state, self.device)
        modified_state = merge_obs_goal(latent_state)
        return super().forward(modified_state)


class VisualGoalConditionedCritic(VisualCritic):
    def forward(self, state, action):
        latent_state = self.encoder.get_latent_state(state, self.device)
        modified_state = merge_obs_goal(latent_state)
        return super().forward(modified_state, action)


class VisionUVFDDPG(nn.Module):
    def __init__(
        self,
        # Encoder args
        width: int,
        height: int,
        in_channels: int,
        embedding_size: int,
        action_dim: int,
        max_action: float,
        act_fn,
        device: torch.device,
        ActorCls=VisualGoalConditionedActor,
        CriticCls=VisualGoalConditionedCritic,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-5,
        tau: float = 0.05,
        discount: float = 1.0,
        ensemble_size: int = 2,
        num_bins: int = 20,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        targets_update_interval: int = 5,
        actor_update_interval: int = 1,
        use_distributional_rl: bool = True,
        # Policy args
    ):
        super(VisionUVFDDPG, self).__init__()
        self.state_dim = embedding_size * 2
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.ensemble_size = ensemble_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_distributional_rl = use_distributional_rl
        self.num_bins = num_bins
        self.targets_update_interval = targets_update_interval
        self.actor_update_interval = actor_update_interval

        if self.use_distributional_rl:
            self.discount = 1
            CriticCls = functools.partial(CriticCls, output_dim=self.num_bins)

        self.actor = ActorCls(
            width=width,
            height=height,
            state_dim=embedding_size * 2,
            action_dim=action_dim,
            max_action=max_action,
            embedding_size=embedding_size,
            act_fn=act_fn,
            in_channels=in_channels,
            encoder=encoder,
            device=device,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-07
        )

        self.critic = CriticCls(
            width=width,
            height=height,
            state_dim=embedding_size * 2,
            action_dim=action_dim,
            embedding_size=embedding_size,
            act_fn=act_fn,
            in_channels=in_channels,
            encoder=encoder,
            device=device,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-07
        )

        if self.ensemble_size > 1:
            self.critic = EnsembledCritic(self.critic, ensemble_size=ensemble_size)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

            for i in range(1, len(self.critic.critics)):  # First copy already added
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group(
                    {"params": critic_copy.parameters()}
                )
                # https://stackoverflow.com/questions/51756913/in-pytorch-how-do-you-use-add-param-group-with-a-optimizer

        self.optimize_iterations = 0

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).cpu().detach().numpy().squeeze()

    def _get_q_values(self, state):
        actions = self.actor(state)
        q_values = self.critic(state, actions)
        return q_values

    def get_q_values(self, state, aggregate="mean"):
        q_values = self._get_q_values(state)
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        expected_q_values_list = []
        if self.use_distributional_rl:
            for q_values in q_values_list:
                q_probs = F.softmax(q_values, dim=1)
                batch_size = q_probs.shape[0]
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                neg_bin_range = -torch.arange(
                    1, self.num_bins + 1, dtype=torch.float
                ).to(q_values.device)
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat(batch_size, 1)
                assert q_probs.shape == tiled_bin_range.shape
                # Take the inner product between these two tensors
                expected_q_values = torch.sum(
                    q_probs * tiled_bin_range, dim=1, keepdim=True
                )
                expected_q_values_list.append(expected_q_values)
        else:
            expected_q_values_list = q_values_list

        expected_q_values = torch.stack(expected_q_values_list)
        if aggregate is not None:
            if aggregate == "mean":
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == "min":
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            else:
                raise ValueError

        if not self.use_distributional_rl:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_q_value = -1.0 * self.num_bins
            max_q_value = 0.0
            expected_q_values = torch.clamp(expected_q_values, min_q_value, max_q_value)

        return expected_q_values

    def optimize(self, replay_buffer: VisualReplayBuffer, iterations=1, batch_size=128):
        opt_info = dict(actor_loss=[], critic_loss=[])
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            state = inp_to_torch_device(state, self.device)
            next_state = inp_to_torch_device(next_state, self.device)
            action = inp_to_torch_device(action, self.device)
            reward = inp_to_torch_device(reward, self.device)
            done = inp_to_torch_device(done, self.device)

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info["critic_loss"].append(critic_loss.cpu().detach().numpy())

            if self.optimize_iterations % self.actor_update_interval == 0:
                # Compute actor loss
                actor_loss = -self.get_q_values(state).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info["actor_loss"].append(actor_loss.cpu().detach().numpy())

            # Update the frozen target models
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()

        return opt_info

    def update_actor_target(self):
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update_critic_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state["observation"]),
                goal=torch.FloatTensor(state["goal"]),
            )
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)

    def critic_loss(self, current_q, target_q, reward, done):
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            if self.use_distributional_rl:
                # Compute distributional td targets
                target_q_probs = F.softmax(target_q, dim=1)
                batch_size = target_q_probs.shape[0]
                one_hot = torch.zeros(batch_size, self.num_bins).to(reward.device)
                one_hot[:, 0] = 1

                # Calculate the shifted probabilities
                # Fist column: Since episode didn't terminate, probability that the
                # distance is 1 equals 0.
                col_1 = torch.zeros((batch_size, 1)).to(reward.device)
                # Middle columns: Simply the shifted probabilities.
                col_middle = target_q_probs[:, :-2]
                # Last column: Probability of taking at least n steps is sum of
                # last two columns in unshifted predictions:
                col_last = torch.sum(target_q_probs[:, -2:], dim=1, keepdim=True)
                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=1)
                assert one_hot.shape == shifted_target_q_probs.shape
                td_targets = torch.where(
                    done.bool(), one_hot, shifted_target_q_probs
                ).detach()

                critic_loss = torch.mean(
                    -torch.sum(td_targets * torch.log_softmax(current_q, dim=1), dim=1)
                )  # https://github.com/tensorflow/tensorflow/issues/21271
            else:
                raise NotImplementedError("Not implemented for non-distributional RL")
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def state_dict(self):
        out = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        return out

    def load_state_dict(self, other: dict):
        self.actor.load_state_dict(other["actor"])
        self.actor.to(device=self.device)

        self.actor_target.load_state_dict(other["actor_target"])
        self.actor_target.to(device=self.device)

        self.critic.load_state_dict(other["critic"])
        self.critic.to(device=self.device)

        self.critic_target.load_state_dict(other["critic_target"])
        self.critic_target.to(device=self.device)

        self.actor_optimizer.load_state_dict(other["actor_optimizer"])
        self.critic_optimizer.load_state_dict(other["critic_optimizer"])

    def get_pairwise_dist(
        self,
        obs_vec,
        goal_vec=None,
        aggregate: Union[str, None] = "mean",
        max_search_steps=7,
        masked=False,
    ):
        """
        Estimates the pairwise distances.

        obs_vec: Array containing observations
        goal_vec: (optional) Array containing a second set of observations. If
                  not specified, computes the pairwise distances between obs_tensor and
                  itself.
        aggregate: (str) How to combine the predictions from the ensemble. Options
                   are to take the minimum predicted q value (i.e., the maximum distance),
                   the mean, or to simply return all the predictions.
        max_search_steps: (int)
        masked: (bool) Whether to ignore edges that are too long, as defined by
                max_search_steps.
        """
        if goal_vec is None:
            goal_vec = obs_vec

        dist_matrix = []
        for obs_index in range(len(obs_vec)):
            obs = obs_vec[obs_index]
            obs_repeat_tensor = np.repeat([obs], len(goal_vec), axis=0)
            state = {"observation": obs_repeat_tensor, "goal": goal_vec}
            dist = self.get_dist_to_goal(state, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = np.stack(dist_matrix)
        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])

        if masked:
            mask = pairwise_dist > max_search_steps
            return np.where(mask, np.full(pairwise_dist.shape, np.inf), pairwise_dist)
        else:
            return pairwise_dist


class LagVisionUVFDDPG(VisionUVFDDPG):
    def __init__(
        self,
        # Encoder args
        width: int,
        height: int,
        in_channels: int,
        embedding_size: int,
        action_dim: int,
        max_action: float,
        act_fn,
        device: torch.device,
        ActorCls=VisualGoalConditionedActor,
        CriticCls=VisualGoalConditionedCritic,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-5,
        tau: float = 0.05,
        discount: float = 1.0,
        ensemble_size: int = 2,
        num_bins: int = 20,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        targets_update_interval: int = 5,
        actor_update_interval: int = 1,
        use_distributional_rl: bool = True,
        cost_kwargs: Optional[dict] = None,
    ):
        """
        # Cost configs
        cost_min:float = 0,
        cost_max:float = 2.0,
        cost_N=20,
        cost_critic_lr:float=1e-3,
        cost_limit:float=1.0,
        lambda_lr:float=0.001,
        lambda_optimizer:str="Adam", # lambda optimizer is actually unused
        """
        super(LagVisionUVFDDPG, self).__init__(
            # Encoder args
            width=width,
            height=height,
            in_channels=in_channels,
            embedding_size=embedding_size,
            action_dim=action_dim,
            max_action=max_action,
            act_fn=act_fn,
            device=device,
            ActorCls=ActorCls,
            CriticCls=CriticCls,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            discount=discount,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            encoder=encoder,
            targets_update_interval=targets_update_interval,
            actor_update_interval=actor_update_interval,
            use_distributional_rl=use_distributional_rl,
        )
        # For lagrangian
        if cost_kwargs:
            self.constraints = cost_kwargs
            self.lagrange = Lagrange(
                cost_limit=cost_kwargs["cost_limit"],
                lagrangian_multiplier_init=cost_kwargs["lagrangian_multiplier_init"],
                lambda_lr=cost_kwargs["lambda_lr"],
                lambda_optimizer=cost_kwargs["lambda_optimizer"],
            )
            self.lagrange_on = cost_kwargs["lagrange_on"]

            self.cost_critic = CriticCls(
                width=width,
                height=height,
                state_dim=embedding_size * 2,
                action_dim=action_dim,
                embedding_size=embedding_size,
                act_fn=act_fn,
                in_channels=in_channels,
                encoder=encoder,
                output_dim=cost_kwargs["cost_N"],
                device=device,
            )
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(),
                lr=cost_kwargs["cost_critic_lr"],
                eps=1e-07,
            )

            self.F_categorical = CategoricalActivation(
                vmin=cost_kwargs["cost_min"],
                vmax=cost_kwargs["cost_max"],
                N=cost_kwargs["cost_N"],
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

    def optimize(
        self, replay_buffer: ConstrainedVisualReplayBuffer, iterations=1, batch_size=128
    ):
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
                actor_loss = 0.0
                if self.lagrange_on:
                    actor_loss_c = self.get_cost_q_values(state)
                    # TODO: Figure out whether the masked version is better, it should enforce a <= constraint?
                    mask_loss_c = actor_loss_c > self.lagrange.cost_limit
                    lag = self.lagrange.lagrangian_multiplier.item()
                    masked_actor_loss_c = (actor_loss_c * mask_loss_c).mean() * lag
                    actor_loss = (actor_loss_r.mean() + masked_actor_loss_c) / (1 + lag)
                else:
                    actor_loss = actor_loss_r.mean()  # debug training, drop lagrange

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

    def update_cost_critic_target(self):
        for param, target_param in zip(
            self.cost_critic.parameters(), self.cost_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def set_lag_status(self, turn_on_lag: bool):
        self.lagrange_on = turn_on_lag

    def cost_critic_loss(
        self,
        current_q: Union[torch.Tensor, List[torch.Tensor]],
        target_q: Union[torch.Tensor, List[torch.Tensor]],
        cost: torch.Tensor,  # (N, 1)
        done: torch.Tensor,  # (N, 1)
    ):
        """
        Loss on cumulative costs
        current_q: A torch tensor if the cost critic is not an ensemble, or a list of torch tensors if the cost critic
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
        max_search_steps: (int)
        masked: (bool) Whether to ignore edges that are too long, as defined by
                max_search_steps.
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
            "actor_target",
            "actor_optimizer",
            "critic",
            "critic_target",
            "critic_optimizer",
        ]
        unconstrained_state_dict = {}
        for key in unconstrained_keys:
            unconstrained_state_dict[key] = state_dict[key]
        super().load_state_dict(unconstrained_state_dict)

        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic_optimizer.load_state_dict(state_dict["cost_critic_optimizer"])


class ConstrainedVisionUVFDDPG(VisionUVFDDPG):
    """
    Should be the parent class of LagVisionUVFDDPG, but don't want to re-run the experiments again
    """

    def __init__(
        self,
        # Encoder args
        width: int,
        height: int,
        in_channels: int,
        embedding_size: int,
        action_dim: int,
        max_action: float,
        act_fn,
        device: torch.device,
        ActorCls=VisualGoalConditionedActor,
        CriticCls=VisualGoalConditionedCritic,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-5,
        tau: float = 0.05,
        discount: float = 1.0,
        ensemble_size: int = 2,
        num_bins: int = 20,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        targets_update_interval: int = 5,
        actor_update_interval: int = 1,
        use_distributional_rl: bool = True,
        cost_kwargs: Optional[dict] = None,
    ):
        """
        # Cost configs
        cost_min:float = 0,
        cost_max:float = 2.0,
        cost_N=20,
        cost_critic_lr:float=1e-3,
        cost_limit:float=1.0,
        lambda_lr:float=0.001,
        lambda_optimizer:str="Adam", # lambda optimizer is actually unused
        """
        super(ConstrainedVisionUVFDDPG, self).__init__(
            # Encoder args
            width=width,
            height=height,
            in_channels=in_channels,
            embedding_size=embedding_size,
            action_dim=action_dim,
            max_action=max_action,
            act_fn=act_fn,
            device=device,
            ActorCls=ActorCls,
            CriticCls=CriticCls,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            discount=discount,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            encoder=encoder,
            targets_update_interval=targets_update_interval,
            actor_update_interval=actor_update_interval,
            use_distributional_rl=use_distributional_rl,
        )

        # For lagrangian
        if cost_kwargs is not None:
            self.cost_critic = CriticCls(
                width=width,
                height=height,
                state_dim=embedding_size * 2,
                action_dim=action_dim,
                embedding_size=embedding_size,
                act_fn=act_fn,
                in_channels=in_channels,
                encoder=encoder,
                output_dim=cost_kwargs["cost_N"],
                device=device,
            )
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(),
                lr=cost_kwargs["cost_critic_lr"],
                eps=1e-07,
            )

            self.F_categorical = CategoricalActivation(
                vmin=cost_kwargs["cost_min"],
                vmax=cost_kwargs["cost_max"],
                N=cost_kwargs["cost_N"],
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

    def optimize(
        self, replay_buffer: ConstrainedVisualReplayBuffer, iterations=1, batch_size=128
    ):
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
                actor_loss = actor_loss_r.mean()
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

    def update_cost_critic_target(self):
        for param, target_param in zip(
            self.cost_critic.parameters(), self.cost_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def cost_critic_loss(
        self,
        current_q: Union[torch.Tensor, List[torch.Tensor]],
        target_q: Union[torch.Tensor, List[torch.Tensor]],
        cost: torch.Tensor,  # (N, 1)
        done: torch.Tensor,  # (N, 1)
    ):
        """
        Loss on cumulative costs
        current_q: A torch tensor if the cost critic is not an ensemble, or a list of torch tensors if the cost critic
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

    def get_pairwise_cost(self, obs_vec, goal_vec=None, aggregate="mean"):
        """
        Estimates the pairwise costs. Return ensemble_size, obs_vec, goal_vec

        obs_vec: Array containing observations
        goal_vec: (optional) Array containing a second set of observations. If
                  not specified, computes the pairwise distances between obs_tensor and
                  itself.
        aggregate: (str) How to combine the predictions from the ensemble. Options
                   are to take the minimum predicted q value (i.e., the maximum distance),
                   the mean, or to simply return all the predictions.
        max_search_steps: (int)
        masked: (bool) Whether to ignore edges that are too long, as defined by
                max_search_steps.
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
            "actor_target",
            "actor_optimizer",
            "critic",
            "critic_target",
            "critic_optimizer",
        ]
        unconstrained_state_dict = {}
        for key in unconstrained_keys:
            unconstrained_state_dict[key] = state_dict[key]
        super().load_state_dict(unconstrained_state_dict)

        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic_optimizer.load_state_dict(state_dict["cost_critic_optimizer"])


class GCOVisionUVFDDPG(ConstrainedVisionUVFDDPG):
    def __init__(
        self,
        # Encoder args
        width: int,
        height: int,
        in_channels: int,
        embedding_size: int,
        action_dim: int,
        max_action: float,
        act_fn,
        device: torch.device,
        ActorCls=VisualGoalConditionedActor,
        CriticCls=VisualGoalConditionedCritic,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-5,
        tau: float = 0.05,
        discount: float = 1.0,
        ensemble_size: int = 2,
        num_bins: int = 20,
        encoder: Literal["VisualEncoder", "VisualRGBEncoder"] = "VisualEncoder",
        targets_update_interval: int = 5,
        actor_update_interval: int = 1,
        use_distributional_rl: bool = True,
    ):
        super(GCOVisionUVFDDPG, self).__init__(
            width=width,
            height=height,
            in_channels=in_channels,
            embedding_size=embedding_size,
            action_dim=action_dim,
            max_action=max_action,
            act_fn=act_fn,
            device=device,
            ActorCls=ActorCls,
            CriticCls=CriticCls,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            discount=discount,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            encoder=encoder,
            targets_update_interval=targets_update_interval,
            actor_update_interval=actor_update_interval,
            use_distributional_rl=use_distributional_rl,
        )

    def optimize(
        self, replay_buffer: ConstrainedVisualReplayBuffer, iterations=1, batch_size=128
    ):
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
                actor_loss = 0.0
                if self.lagrange_on:
                    actor_loss_c = self.get_cost_q_values(state)
                    # TODO: Figure out whether the masked version is better, it should enforce a <= constraint?
                    mask_loss_c = actor_loss_c > self.lagrange.cost_limit  # type: ignore
                    lag = self.lagrange.lagrangian_multiplier.item()  # type: ignore
                    masked_actor_loss_c = (actor_loss_c * mask_loss_c).mean() * lag
                    actor_loss = (actor_loss_r.mean() + masked_actor_loss_c) / (1 + lag)
                else:
                    actor_loss = actor_loss_r.mean()  # debug training, drop lagrange

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
