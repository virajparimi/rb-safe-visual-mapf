import torch
import numpy as np
from torch import nn
from typing import Union
from termcolor import colored

from pud.algos.ddpg import UVFDDPG
from pud.utils import variance_initializer_
from pud.algos.data_struct import inp_to_torch_device


class Encoder(nn.Module):
    """
    Encoder parent class, must have goal-conditioned get_latent_state
    """

    def __init__(self, device: torch.device = torch.device("cpu"), **kwargs):
        super(Encoder, self).__init__()
        self.device = device

    def get_latent_state(
        self, image_state: Union[torch.Tensor, dict, np.ndarray], device: torch.device
    ):
        """Convert image state to latent state"""
        inp = inp_to_torch_device(image_state, device=device)  # type: ignore

        if torch.is_tensor(inp):
            return self.forward(inp)
        elif isinstance(inp, dict):
            if len(inp["observation"].shape) == 5:
                # Batch is constructed on dim 0,
                # e.g., batch_size, 4, 64, 64, 4
                # Combine batch_size and 4 (first dim) for batch computation
                batch_size, num_images, width, height, num_channels = inp[
                    "observation"
                ].shape
                obs_cat = inp["observation"].reshape(
                    [batch_size * num_images, width, height, num_channels]
                )
                goal_cat = inp["goal"].reshape(
                    [batch_size * num_images, width, height, num_channels]
                )
                inp["observation"] = obs_cat  # type: ignore
                inp["goal"] = goal_cat  # type: ignore
            latent_obs = self.forward(inp["observation"].float())  # type: ignore # float32
            latent_goal = self.forward(inp["goal"].float())  # type: ignore # float32
            latent_state = {}
            for key in inp:
                if key == "observation":
                    latent_state[key] = latent_obs
                elif key == "goal":
                    latent_state[key] = latent_goal
                else:
                    latent_state[key] = inp[key]
            return latent_state


class VisualRGBEncoder(Encoder):
    def __init__(
        self,
        in_channels: int = 4,
        embedding_size: int = 256,
        width: int = 32,
        height: int = 32,
        act_fn=nn.SELU,
        ngroups: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super(VisualRGBEncoder, self).__init__(device=device)
        self.width = width
        self.height = height

        self.nets = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            act_fn(),
            nn.GroupNorm(num_groups=ngroups, num_channels=16),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.GroupNorm(num_groups=ngroups, num_channels=32),
            nn.Flatten(),
        ]

        # Lazy initialization
        mlp_inp_dim = self.lazy_init()
        self.emb_mlp = nn.Linear(int(mlp_inp_dim), embedding_size).to(
            device=self.device
        )
        for layer in self.nets:
            layer.to(device=self.device)

    def lazy_init(self):
        with torch.no_grad():
            dummpy_inp = torch.rand(4, self.height, self.width, 4)
            out = self.encode(dummpy_inp)
            return np.prod(list(out.shape))

    def encode(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # batch_dim, channel_dim, *image_size
        x = x / 255.0
        out = x
        for layer in self.nets:
            out = layer(out)
        return out

    def forward(self, x: torch.Tensor):
        out = self.encode(x)
        num_cat_channels, _ = out.shape
        assert num_cat_channels % 4 == 0
        batch_size = int(num_cat_channels / 4)
        out = out.reshape(batch_size, -1)
        return self.emb_mlp(out)

    def reset_parameters(self):
        variance_initializer_(
            self.emb_mlp.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.emb_mlp.bias)


class VisualEncoder(Encoder):
    """
    Conv are performed on individual images (4 directions)
    -> image embeddings
    state embedding <- MLP(4 x image embeddings)
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_size: int = 256,
        width: int = 32,
        height: int = 32,
        act_fn=nn.SELU,
        device: torch.device = torch.device("cpu"),
    ):
        super(VisualEncoder, self).__init__(device=device)

        self.conv_net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16, kernel_size=8, stride=4
            ),  # 64x64 -> 15x15
            act_fn(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4),  # 15x15 -> 3x3
            act_fn(),
            nn.Flatten(),
        )
        # Lazy calculate embedding input size
        l1_inp_size = None
        with torch.no_grad():
            img_size = (width, height)
            tmp = torch.zeros(4, 4, *img_size)
            tmp_out = self.conv_net(tmp)
            l1_inp_size = np.prod(list(tmp_out.shape))

        # 4 direction obs in batch dim x embedding size
        self.l1 = nn.Linear(int(l1_inp_size), embedding_size)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # batch_dim, channel_dim, *image_size
        x = x / 255.0
        out = self.conv_net(x)
        num_cat_channels, _ = out.shape
        assert num_cat_channels % 4 == 0
        batch_size = int(num_cat_channels / 4)
        out = out.reshape(batch_size, -1)
        out = self.l1(out)
        return out

    def reset_parameters(self):
        for i in range(len(self.conv_net)):
            if hasattr(self.conv_net[i], "weight"):
                torch.nn.init.xavier_uniform_(torch.Tensor(self.conv_net[i].weight))
        variance_initializer_(
            self.l1.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l1.bias)


class VisualDecoder(nn.Module):
    def __init__(self, emb_size: int = 256):
        super(VisualDecoder, self).__init__()

        self.l_emb = nn.Linear(emb_size, 4 * 32 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=4,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=4,
            kernel_size=8,
            stride=4,
        )

    def forward(self, emb: torch.Tensor):
        batch_size, emb_dim = emb.shape
        out = self.l_emb(emb)
        out = out.reshape([batch_size * 4, 32, 3, 3])
        out = self.deconv1(out, output_size=torch.Size([4, 16, 15, 15]))
        out = self.deconv2(out, output_size=[batch_size * 4, 4, 64, 64])
        # batch_dim, channel_dim, image size 1, image size 2
        # to
        # batch_dim, image size 1, image size 2, channel dim
        out = out.permute(0, 2, 3, 1)  # batch_dim, channel_dim, *image_size
        return out

    def reset_parameters(self):
        variance_initializer_(
            self.l_emb.weight, scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        torch.nn.init.zeros_(self.l_emb.bias)
        torch.nn.init.xavier_uniform_(self.deconv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)


class VisualUVFDDPG(UVFDDPG):
    def __init__(
        self,
        # Encoder args
        in_channels: int,
        embedding_size: int,
        act_fn,
        device: str,
        # Policy args
        uvfddpg_kwargs: dict,
    ):
        super(VisualUVFDDPG, self).__init__(**uvfddpg_kwargs)
        self.encoder = VisualEncoder(
            in_channels=in_channels,
            embedding_size=embedding_size,
            act_fn=act_fn,
        )
        self.device = torch.device(device)

    def load_pretrained_encoder(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        self.encoder.load_state_dict(ckpt.state_dict())
        print("[{}] Loaded ckpt: {}".format(colored("success", color="green"), ckpt))

    def select_action(self, state):
        latent_state = self.encoder.get_latent_state(state, device=self.device)
        with torch.no_grad():
            return self.actor(latent_state).cpu().detach().numpy().flatten()

    def get_q_values(self, state, aggregate="mean"):
        latent_state = self.encoder.get_latent_state(state, device=self.device)
        return super().get_q_values(latent_state, aggregate)

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            if isinstance(state["observation"], list):
                state["observation"] = np.stack(state["observation"])
                state["goal"] = np.stack(state["goal"])
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)

    def optimize(self, replay_buffer, iterations=1, batch_size=128):
        opt_info = dict(actor_loss=[], critic_loss=[])
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            latent_state = self.encoder.get_latent_state(state, device=self.device)
            next_latent_state = self.encoder.get_latent_state(
                next_state, device=self.device
            )

            action = inp_to_torch_device(action, self.device)
            reward = inp_to_torch_device(reward, self.device)
            done = inp_to_torch_device(done, self.device)

            current_q = self.critic(latent_state, action)
            target_q = self.critic_target(
                next_latent_state, self.actor_target(next_latent_state)
            )
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
