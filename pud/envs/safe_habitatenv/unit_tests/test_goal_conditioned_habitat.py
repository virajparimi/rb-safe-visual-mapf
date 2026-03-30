import torch
import numpy as np
from torch import nn

from pud.algos.vision.visual_models import VisualEncoder
from pud.buffers.visual_buffer import VisualReplayBuffer
from pud.envs.habitat_navigation_env import (
    habitat_env_load_fn,
    GoalConditionedHabitatPointWrapper,
)

device = "cpu"
scene = "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
apsp_path = "pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle"
simulator_settings = dict(
    scene="scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    width=64,
    height=64,
    default_agent=0,
    sensor_height=1.5,
)

env = habitat_env_load_fn(
    height=0,
    device=device,
    action_noise=1.0,
    apsp_path=apsp_path,
    max_episode_steps=20,
    env_type="HabitatSim",
    terminate_on_timeout=False,
    simulator_settings=simulator_settings,
    gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),
)

latent_dimensions = 512
obs_dim = env.observation_space["observation"].shape  # type: ignore
goal_dim = env.observation_space["goal"].shape  # type: ignore
state_dim = (
    latent_dimensions * obs_dim[0] * 2
)  # For each image along cardinal directions and the same for the goal

action_dim = env.action_space.shape[0]  # type: ignore
max_action = float(env.action_space.high[0])  # type: ignore

buffer = VisualReplayBuffer(
    obs_dim=obs_dim,
    goal_dim=goal_dim,
    action_dim=action_dim,
    max_size=1000,
)

reset_result = env.reset()
if len(reset_result) == 2:
    state, info = reset_result
else:
    state = reset_result
obs = state["observation"]

oo1 = torch.rand(12, 64, 64, 4).float()
oo2 = torch.rand(4, 64, 64, 4).float()

e1 = VisualEncoder()
e1 = e1.float()
tmp = e1(oo1)

e1.get_latent_state(state, torch.device("cpu"))

emb = e1(state)
emb.shape

rtol = 1e-4
with torch.no_grad():
    eq1 = torch.allclose(e1(oo1)[0:1], e1(oo1[:4]), rtol=rtol)

    eq2 = torch.allclose(e1(oo1)[1:2], e1(oo1[4:8]), rtol=rtol)

    eq3 = torch.allclose(e1(oo1)[2:3], e1(oo1[8:]), rtol=rtol)

    eq4 = torch.allclose(e1(oo1)[1:2], e1(oo2), rtol=rtol)

    print(eq1, eq2, eq3, eq4)


e1(oo2)
emb = e1(oo1)

# Make sure the embedding is image-specific, even for goal-conditioned encoders
input_channels = 4
conv1 = nn.Sequential(
    nn.Conv2d(
        in_channels=input_channels, out_channels=16, kernel_size=8, stride=4
    ),  # 64x64 -> 15x15
)

conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=4)  # 15x15 -> 3x3

obs_t = oo1
out1_1 = conv1(obs_t)
out1_1.shape  # 4, 16, 15, 15
out1_1_2 = conv2(out1_1)

l1_size = list(out1_1_2.shape)[:]  # 4, 32, 3, 3
l1 = nn.Linear(int(np.prod(l1_size)), 256)

out1_2 = torch.flatten(out1_1_2, start_dim=1)
out1_2r = out1_2.reshape(2, 4 * 288)
out1_2.shape

torch.allclose(torch.flatten(out1_2[:4]), out1_2r[0])
torch.allclose(torch.flatten(out1_2[4:]), out1_2r[1])


# Decode
batch_size, emb_dim = emb.shape

l2 = nn.Linear(256, 4 * 32 * 3 * 3)
out2 = l2(emb)

out2_1 = out2.reshape([batch_size * 4, 32, 3, 3])
out2_1.shape

ctn1 = nn.ConvTranspose2d(
    in_channels=32,
    out_channels=16,
    kernel_size=4,
    stride=4,
)
out2_2 = ctn1(out2_1, output_size=torch.Size([4, 16, 15, 15]))
out2_2.shape

ctn2 = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=4,
    kernel_size=8,
    stride=4,
)

out2_3 = ctn2(out2_2, output_size=[-1, 4, 64, 64])
out2_3.shape
