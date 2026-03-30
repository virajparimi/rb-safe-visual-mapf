import time
import torch
import hydra
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl._utils import logger as torchrl_logger
from torchrl.modules import (
    AdditiveGaussianWrapper,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
)
from torchrl.envs import (
    check_env_specs,
    ExplorationType,
    set_exploration_type,
    TransformedEnv,
    StepCounter,
)

from pud.envs.torchrl_navigation_env import MultiAgentPointEnv
from pud.torchrl_logging import init_logging, extract_episodic_records


def load_checkpoint(
    checkpoint_path,
    policy,
    exploration_policy,
):
    torchrl_logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["policy"])
    exploration_policy.load_state_dict(checkpoint["exploration_policy"])
    return policy, exploration_policy


@hydra.main(
    version_base="1.1",
    config_path="../../configs",
    config_name="config_IDDPG_PointEnv",
)
def eval(cfg):

    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device

    torch.manual_seed(cfg.seed)

    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters

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
    )

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
    policy_net = MultiAgentMLP(
        n_agent_inputs=transformedEvalEnv.observation_spec["agents", "observation"].shape[
            -1
        ]
        * 2,  # Concatenated state and goal
        n_agent_outputs=transformedEvalEnv.full_action_spec["agents", "action"].shape[-1],
        n_agents=cfg.env.num_agents,
        centralised=False,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    policy_mod = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation", "goal_conditioned_state")],
        out_keys=[("agents", "param")],
    )
    policy_module = TensorDictSequential(gc_module, policy_mod)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=transformedEvalEnv.full_action_spec["agents", "action"],
        in_keys=[("agents", "param")],
        out_keys=[("agents", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": transformedEvalEnv.full_action_spec["agents", "action"].space.low,  # type: ignore
            "max": transformedEvalEnv.full_action_spec["agents", "action"].space.high,  # type: ignore
        },
        return_log_prob=False,
    )

    exploration_policy = AdditiveGaussianWrapper(
        policy=policy,
        annealing_num_steps=int(cfg.collector.total_frames * (1 / 2)),
        action_key=("agents", "action"),
    )

    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "DDPG"
        )
        logger = init_logging(cfg, model_name)

    assert cfg.eval.ckpt_path, "Please provide a checkpoint path for evaluation"

    policy, exploration_policy = load_checkpoint(cfg.eval.ckpt_path, policy, exploration_policy)

    evaluation_start = time.time()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        transformedEvalEnv.frames = []
        rollouts = transformedEvalEnv.rollout(
            cfg.env.max_steps * 5,
            policy=policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
    evaluation_time = time.time() - evaluation_start
    extract_episodic_records(logger, rollouts, transformedEvalEnv, evaluation_time)  # type: ignore


if __name__ == "__main__":
    eval()
