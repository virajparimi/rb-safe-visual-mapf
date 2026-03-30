import unittest
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from pud.envs.safe_habitatenv.safe_habitatenv import SafeHabitatNavigationEnv
from pud.envs.habitat_navigation_env import plot_wall, get_default_habitat_sim_settings

"""
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.reset
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.step
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.cost_map
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.one_cost_contour
"""

scenes = []
for scene_id in range(0, 5):
    for stage_id in range(0, 21):
        scenes.append("sc{}_staging_{:0>2d}".format(scene_id, stage_id))


def cost_contour(scene_name: str, normalize=True):
    sim_settings = get_default_habitat_sim_settings("ReplicaCAD")
    sim_settings["scene"] = scene_name
    env = SafeHabitatNavigationEnv(
        env_type="ReplicaCAD",
        sensor_type="rgb",
        simulator_settings=sim_settings,
        device="cuda:1",
        cost_f_args={"name": "linear", "radius": 1},
        cost_limit=1,
    )

    x = np.linspace(0, env.wall_height + 1, int(2 * (env.wall_height + 1)), dtype=float)
    y = np.linspace(0, env.wall_width + 1, int(2 * (env.wall_width + 1)), dtype=float)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * np.inf

    pbar = tqdm(total=X.shape[0], desc="Outer loop")
    assert env.cost_function is not None
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            min_d, _ = env.dist_2_blocks(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = env.cost_function(min_d)
        pbar.update()
    pbar.close()

    fig, ax = plt.subplots()
    if normalize:
        CS = ax.contour(
            X / float(env.wall_height),
            Y / float(env.wall_width),
            Z,
            levels=[0, 0.25, 0.5, 1, 2],
        )
    else:
        CS = ax.contour(X, Y, Z, levels=[0, 0.25, 0.5, 1, 2])
    labels = ax.clabel(CS, inline=False, fontsize=8)

    # Position labels outside the plot
    for label in labels:
        _ = label.get_position()

    ax = plot_wall(walls=env.walls, ax=ax, normalize=normalize)
    ax.set_title("Test Safe HabitatEnv Cost Contour: {}".format(scene_name))
    fig.savefig("runs/tmp_plots/test_cost_contour_{}.jpg".format(scene_name), dpi=300)
    plt.close(fig=fig)
    del env


class TestSafeHabitatEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = dict(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=1.0,
        )

    def reset(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "linear", "radius": 3.0},
            cost_limit=10.0,
        )
        env.reset()

    def step(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=10.0,
        )
        s0, info = env.reset()  # type: ignore
        action = env.action_space.sample()
        env.step(action)

    def cost_map(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "linear", "radius": 0.5},
            cost_limit=0.0,
        )
        fig, ax = plt.subplots()
        plot_wall(walls=env.walls, ax=ax)

        if env.cost_map is not None:
            unsafe_points = np.where(env.cost_map > env.cost_limit)
            unsafe_points = np.column_stack(unsafe_points)
            ax.scatter(
                unsafe_points[:, 0] / float(env.wall_height),
                unsafe_points[:, 1] / float(env.wall_width),
                s=2,
                marker="o",
                c="red",
            )
        ax.set_title("TestSafeHabitatEnv test cost_map")
        fig.savefig(fname="runs/tmp_plots/test_plot_cost_map.jpg", dpi=300)
        plt.close(fig)

    def one_cost_contour(self):
        scene_name = "sc0_staging_20"
        scene_name = "sc3_staging_05"
        scene_name = "sc3_staging_11"
        scene_name = "sc3_staging_15"

        print(scene_name)
        cost_contour(scene_name, normalize=False)

    def all_cost_contour(self):
        pbar = tqdm(total=len(scenes))
        for scr in scenes:
            try:
                cost_contour(scr)
            except Exception:
                print("Failed: {}".format(scr))
            pbar.update()


if __name__ == "__main__":
    unittest.main()
