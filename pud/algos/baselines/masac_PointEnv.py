import time

import torch
import hydra
from torch import nn

from tensordict import unravel_key
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.envs import Transform
from torchrl.data import TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl._utils import logger as torchrl_logger
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import SACLoss, SoftUpdate, ValueEstimators
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, StepCounter

from pud.envs.torchrl_navigation_env import MultiAgentPointEnv
from pud.torchrl_logging import init_logging, log_evaluation, log_training


def save_checkpoint(
    checkpoint_path,
    collector,
    policy,
    critic,
    optimizer,
    replay_buffer,
    loss_module,
):
    checkpoint = {
        "collector": collector.state_dict(),
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_buffer": replay_buffer.state_dict(),
        "loss_module": loss_module.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    torchrl_logger.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(
    checkpoint_path,
    collector,
    policy,
    critic,
    optimizer,
    replay_buffer,
    loss_module,
):
    torchrl_logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    collector.load_state_dict(checkpoint["collector"])
    policy.load_state_dict(checkpoint["policy"])
    critic.load_state_dict(checkpoint["critic"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    loss_module.load_state_dict(checkpoint["loss_module"])
    return (
        collector,
        policy,
        critic,
        optimizer,
        replay_buffer,
        loss_module,
    )


def swap_last(source, dest):
    source = unravel_key(source)
    dest = unravel_key(dest)
    if isinstance(source, str):
        if isinstance(dest, str):
            return dest
        return dest[-1]
    if isinstance(dest, str):
        return source[:-1] + (dest,)
    return source[:-1] + (dest[-1],)


class DoneTransform(Transform):
    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                tensordict.get(("next", done_key))
                .unsqueeze(-1)
                .expand(tensordict.get(("next", self.reward_key)).shape),
            )
        return tensordict


@hydra.main(
    version_base="1.1",
    config_path="../../../configs/baseline_configs",
    config_name="config_MASAC_PointEnv",
)
def train(cfg):

    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device

    torch.manual_seed(cfg.seed)

    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters

    env = MultiAgentPointEnv(
        walls=cfg.env.walls,
        num_agents=cfg.env.num_agents,
        batch_size=num_envs,
        seed=cfg.seed,
        resize_factor=cfg.env.resize_factor,
        thin=cfg.env.thin,
        action_noise=cfg.env.action_noise,
        apsp_path=cfg.env.apsp_path,
        device=cfg.env.device,
        reward_type=cfg.env.reward_type,
    )

    eval_env = MultiAgentPointEnv(
        walls=cfg.env.walls,
        num_agents=cfg.env.num_agents,
        batch_size=cfg.eval.evaluation_episodes,
        seed=cfg.seed,
        resize_factor=cfg.env.resize_factor,
        thin=cfg.env.thin,
        action_noise=cfg.env.action_noise,
        apsp_path=cfg.env.apsp_path,
        device=cfg.env.device,
        reward_type=cfg.env.reward_type,
    )

    transformedEnv = TransformedEnv(env, StepCounter(max_steps=cfg.env.max_steps))
    check_env_specs(transformedEnv)
    transformedEnv = TransformedEnv(
        transformedEnv,
        RewardSum(
            in_keys=transformedEnv.reward_keys,
            reset_keys=["_reset"] * len(transformedEnv.group_map.keys()),
        ),
    )
    check_env_specs(transformedEnv)

    # 100 steps for more time for the agent to find the goal
    transformedEvalEnv = TransformedEnv(
        eval_env, StepCounter(max_steps=cfg.env.max_steps * 5)
    )
    check_env_specs(transformedEvalEnv)

    gc_module = TensorDictModule(
        lambda state, goal: torch.cat([state, goal], dim=-1),
        in_keys=[("agents", "observation", "state"), ("agents", "observation", "goal")],
        out_keys=[("agents", "observation", "goal_conditioned_state")],
    )
    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=transformedEnv.observation_spec["agents", "observation"].shape[
                -1
            ]
            * 2,  # Concatenated state and goal
            n_agent_outputs=2 * transformedEnv.full_action_spec["agents", "action"].shape[-1],
            n_agents=cfg.env.num_agents,
            centralised=False,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_mod = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation", "goal_conditioned_state")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy_module = TensorDictSequential([gc_module, policy_mod])

    policy = ProbabilisticActor(
        module=policy_module,
        spec=transformedEnv.full_action_spec["agents", "action"],
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": transformedEnv.full_action_spec["agents", "action"].space.low,  # type: ignore
            "max": transformedEnv.full_action_spec["agents", "action"].space.high,  # type: ignore
        },
        return_log_prob=True,
    )

    gc_module = TensorDictModule(
        lambda state, goal: torch.cat([state, goal], dim=-1),
        in_keys=[("agents", "observation", "state"), ("agents", "observation", "goal")],
        out_keys=[("agents", "observation", "goal_conditioned_state")],
    )

    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[
            ("agents", "observation", "goal_conditioned_state"),
            ("agents", "action"),
        ],
        out_keys=[("agents", "obs_action")],
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=(
            transformedEnv.observation_spec["agents", "observation"].shape[-1] * 2
        )
        + transformedEnv.full_action_spec["agents", "action"].shape[-1],
        n_agent_outputs=1,
        n_agents=cfg.env.num_agents,
        centralised=cfg.model.centralised_critic,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "obs_action")],
        out_keys=[("agents", "state_action_value")],
    )

    critic = TensorDictSequential([gc_module, cat_module, critic_module])

    collector = SyncDataCollector(
        transformedEnv,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(
            reward_key=transformedEnv.reward_key, done_keys=transformedEnv.done_keys
        ),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),  # type: ignore
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = SACLoss(
        actor_network=policy,
        qvalue_network=critic,
        delay_value=True,
        action_spec=transformedEnv.full_action_spec["agents", "action"],
    )
    loss_module.set_keys(
        state_action_value=("agents", "state_action_value"),
        action=("agents", "action"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)

    target_updaters = SoftUpdate(loss_module, eps=1 - cfg.loss.tau)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=float(cfg.train.lr))

    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "SAC"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()

    if cfg.train.ckpt_path:
        (
            collector,
            policy,
            critic,
            optimizer,
            replay_buffer,
            loss_module,
        ) = load_checkpoint(
            cfg.train.ckpt_path,
            collector,
            policy,
            critic,
            optimizer,
            replay_buffer,
            loss_module,
        )

    for iteration, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {iteration}")

        sampling_time = time.time() - sampling_start

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):

                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = loss_vals["loss_actor"] + loss_vals["loss_alpha"] + loss_vals["loss_qvalue"]
                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optimizer.step()
                optimizer.zero_grad()
                target_updaters.step()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,  # type: ignore
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                iteration,
                current_frames,
                total_frames,
                step=iteration,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and iteration % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                transformedEvalEnv.frames = []
                rollouts = transformedEvalEnv.rollout(
                    cfg.env.max_steps * 5,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
            evaluation_time = time.time() - evaluation_start
            log_evaluation(
                logger, rollouts, transformedEvalEnv, evaluation_time, step=iteration  # type: ignore
            )

            ckpt_path = (
                logger.log_dir.split(logger.exp_name)[0]
                + "checkpoint_"
                + str(iteration)
                + ".pth"
            )
            save_checkpoint(
                ckpt_path,
                collector,
                policy,
                critic,
                optimizer,
                replay_buffer,
                loss_module,
            )

        sampling_start = time.time()


def rendering_callback(env, td):
    env.frames.append(env._render(td, mode="rgb_array"))


if __name__ == "__main__":
    train()
