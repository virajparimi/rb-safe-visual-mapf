import yaml
import argparse
from pathlib import Path
from dotmap import DotMap
import matplotlib.pyplot as plt


from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls, plot_trajs
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="training configuration",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="pud/envs/safe_pointenv/unit_tests/outputs/grid_trajs",
        help="override ckpt dir",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose printing/logging"
    )
    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)
    cfg.pprint()

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=False,
    )
    assert isinstance(env, SafeGoalConditionedPointBlendWrapper)

    eval_distances = cfg.runner.eval_distances
    cost_intervals = cfg.runner.eval_cost_intervals

    outdir = Path(args.outdir)

    for idx_d in range(len(eval_distances)):
        min_dist, max_dist = eval_distances[idx_d], eval_distances[idx_d]
        for idx_c in range(len(cost_intervals)):
            fig, ax = plt.subplots()
            plot_safe_walls(env.get_map(), env.get_cost_map(), env.cost_limit, ax=ax)
            min_cost, max_cost = cost_intervals[idx_c], cost_intervals[idx_c]

            trajs = []
            for i in range(10):
                out = env.cbfs_sample(
                    min_dist=min_dist,
                    max_dist=max_dist,
                    min_cost=min_cost,
                    max_cost=max_cost,
                )

                traj, traj_cost = out
                trajs.append(traj)

            plot_trajs(trajs, env.get_map(), ax=ax)
            ax.legend()

            outdir.mkdir(parents=True, exist_ok=True)
            outpath = outdir.joinpath(
                "c={:.2f}_d={:0>2d}.jpg".format(min_cost, min_dist)
            )
            fig.savefig(outpath, dpi=300)
            plt.close(fig=fig)
