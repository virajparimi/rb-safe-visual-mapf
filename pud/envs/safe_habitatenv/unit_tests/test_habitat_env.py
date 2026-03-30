import unittest
import numpy as np
import matplotlib.pyplot as plt

from pud.envs.habitat_navigation_env import (
    plot_wall,
    HabitatNavigationEnv,
)


"""
python pud/envs/safe_habitatenv/unit_tests/test_habitat_env.py TestHabitatEnv.compare_occupancy
python pud/envs/safe_habitatenv/unit_tests/test_habitat_env.py TestHabitatEnv.plot_wall
"""


class TestHabitatEnv(unittest.TestCase):

    def test_obs(self):
        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            height=0,
            sensor_type="rgb",
        )

        selected_points = [
            [0.4837957993182139, 0.5812356979405034],
            [0.4679084039817981, 0.5240274599542334],
            [0.31365823772444756, 0.931350114416476],
        ]

        pnts_arr = np.fliplr(np.array(selected_points))

        grid_size = np.array(env._walls.shape)
        grid_pos = pnts_arr * grid_size

        obs_list = [
            env.get_sensor_obs_at_grid_xy(grid_pos[idx])
            for idx in range(len(selected_points))
        ]
        obs_list_2 = [
            env.get_sensor_obs_at_grid_xy(grid_pos[idx])
            for idx in range(len(selected_points))
        ]

        for i in range(len(obs_list)):
            np.allclose(obs_list[i], obs_list_2[i])

    def plot_wall(self):
        scene = "sc0_staging_20"
        scene = "sc2_staging_08"
        scene = "sc3_staging_05"
        scene = "sc3_staging_11"
        scene = "sc3_staging_15"

        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            scene=scene,
            sensor_type="rgb",
            device="cuda:1",
        )
        fig, ax = plt.subplots()
        plot_wall(walls=env.walls, ax=ax, normalize=False)
        ax.set_title("{}, h={}, w={}".format(scene, env.wall_height, env.wall_width))
        fig.savefig("runs/tmp_plots/{}.jpg".format(scene), dpi=300)

    def compare_occupancy(self):
        """Compare the occupancy from the 2D maze matrix and habitat"""
        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            height=0,
        )
        diff_set = []

        h, w = env.walls.shape
        for i in range(h):
            for j in range(w):
                grid_xy = (i, j)
                habitat_xy = env.get_habitat_xy_from_grid_xy(grid_xy)
                # 1 is empty, 0 is blocked
                if env.walls[i, j] != 1 - env.is_blocked_habitat(np.array(habitat_xy)):
                    diff_set.append((i, j))

        fig, ax = plt.subplots()
        ax = plot_wall(env.walls, ax=ax)
        diff_arr = np.array(diff_set)
        ax.scatter(
            diff_arr[:, 1] / w,
            diff_arr[:, 0] / h,
            color="r",
            zorder=2,
            marker="o",
            label="inconsistency",
            s=40,
        )
        ax.legend()
        fig.savefig(fname="runs/tmp_plots/walls.jpg", dpi=300)
        plt.close(fig=fig)


if __name__ == "__main__":
    unittest.main()
