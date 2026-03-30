import unittest
import habitat_sim
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

"""
Mostly copied from habitat-sim/examples/tutorials/nb_python/ReplicaCAD_quickstart.py
"""

"""
python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.plot_map
python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.load_hatbitat_cad
python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.vis_handed_crafted_waypoints  # noqa
python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.vis_handed_crafted_waypoints_w_topdown_maps
"""


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        sim_cfg.override_scene_light_defaults = settings[
            "override_scene_light_defaults"
        ]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]

    # Note: All sensors must have the same resolution
    sensor_specs = []
    color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
    color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
    color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_1st_person_spec.resolution = [
        settings["height"],
        settings["width"],
    ]
    color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_1st_person_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_1st_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_default_settings():
    settings = {
        "width": 1280,  # Spatial resolution of the observations
        "height": 720,
        "scene_dataset": "replica cad dataset path",  # dataset path
        "scene": "sc1_staging_00",  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": 0.0,  # sensor pitch (x rotation in rads)
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
    }
    return settings


class TestReplicaCADBarebone(unittest.TestCase):
    def test_construct_sim(self):
        scene_dataset = (
            "external_data/replica_cad/replica_cad_baked_lighting/"
            "replicaCAD_baked.scene_dataset_config.json"
        )
        settings = make_default_settings()
        settings["scene_dataset"] = scene_dataset

        cfg = make_cfg(settings)
        _ = habitat_sim.Simulator(cfg)

    def vis_handed_crafted_waypoints(self):
        from pud.envs.habitat_navigation_env import HabitatNavigationEnv

        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
        )

        height, width = env._walls.shape
        waypoints = np.loadtxt(
            "pud/envs/safe_habitatenv/unit_tests/waypoints.txt", delimiter=","
        )
        # Waypoints in 2d grid
        waypoints = waypoints * np.array([height, width], dtype=float)
        obs_at_waypoints = [env.get_sensor_obs_at_grid_xy(wp) for wp in waypoints]

        assert env.sensor_type == "rgb"
        pbar = tqdm(total=len(obs_at_waypoints))
        for i_obs, obs_cat in enumerate(obs_at_waypoints):
            fig, ax = plt.subplots(nrows=2, ncols=2)
            for i in range(4):
                ax[i % 2, i // 2].imshow((obs_cat[i]).astype(dtype="uint8"))

            target_dir = Path("temp/trace_bounds/")
            target_dir.mkdir(parents=True, exist_ok=True)
            fig_path = target_dir.joinpath("trace_bounds_{:0>3d}.jpg".format(i_obs))
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pbar.update()

    def vis_handed_crafted_waypoints_w_topdown_maps(self):
        from pud.envs.habitat_navigation_env import HabitatNavigationEnv

        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
        )

        height, width = env._walls.shape
        waypoints = np.loadtxt("runs/tmp_plots/waypoints.txt", delimiter=",")
        waypoints_map = waypoints * np.array([height, width], dtype=float)
        obs_at_waypoints = [env.get_sensor_obs_at_grid_xy(wp) for wp in waypoints_map]

        assert env.sensor_type == "rgb"
        pbar = tqdm(total=len(obs_at_waypoints))
        for i_obs, obs_cat in enumerate(obs_at_waypoints):
            fig = plt.figure(layout="constrained")
            gs = GridSpec(nrows=2, ncols=3, figure=fig)
            for i in range(4):
                axi = fig.add_subplot(gs[i % 2, i // 2])
                axi.imshow((obs_cat[i]).astype(dtype="uint8"))

            ax_wall = fig.add_subplot(gs[:, -1])
            walls = env._walls.copy()
            for i, j in zip(*np.where(walls)):
                x = np.array([i, i + 1]) / float(height)
                y0 = np.array([j, j]) / float(width)
                y1 = np.array([j + 1, j + 1]) / float(width)
                ax_wall.fill_between(x, y0, y1, color="grey")

            ax_wall.set_xlim((0, 1))
            ax_wall.set_ylim((0, 1))
            ax_wall.set_xticks([])
            ax_wall.set_yticks([])
            ax_wall.set_aspect("equal", adjustable="box")

            ax_wall.plot([waypoints[i_obs][0]], [waypoints[i_obs][1]], "ro")
            ax_wall.plot(
                waypoints[:, 0], waypoints[:, 1], color="y", linestyle="--", linewidth=2
            )
            ax_wall.plot(
                waypoints[i_obs][0],
                waypoints[i_obs][1],
                marker="o",
                color="red",
                markersize=6,
                zorder=4,
            )

            target_dir = Path("runs/tmp_plots/trace_bounds/")
            target_dir.mkdir(parents=True, exist_ok=True)
            fig_path = target_dir.joinpath("trace_bounds_{:0>3d}.jpg".format(i_obs))
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pbar.update()

    def plot_map(self):
        """Plot the walls and trajectory"""
        from pud.envs.habitat_navigation_env import (
            HabitatNavigationEnv,
            plot_wall,
            plot_traj,
        )

        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
        )
        height, width = env._walls.shape
        waypoints = np.loadtxt("runs/tmp_plots/waypoints.txt", delimiter=",")

        fig, ax = plt.subplots()
        plot_wall(env.walls, ax=ax)
        waypoints_map = waypoints * np.array([height, width], dtype=float)
        plot_traj(
            waypoints_map,
            walls=env.walls,
            ax=ax,
            color="y",
            linestyle="--",
            linewidth=2,
            normalize=False,
        )
        ax.set_title("Test Case: Plot Map")
        fig.savefig("runs/tmp_plots/test_plot_traj.jpg", dpi=300)

    def load_hatbitat_cad(self):
        import yaml
        from pud.envs.habitat_navigation_env import habitat_env_load_fn

        config_file = "configs/config_HabitatReplicaCAD.yaml"
        habitat_config = {}
        with open(config_file, "r") as f:
            habitat_config = yaml.safe_load(f)
        env_config = habitat_config["env"]

        env = habitat_env_load_fn(
            max_episode_steps=20,
            **env_config,
        )

        self.assertIsNotNone(env, "env should not be None")


if __name__ == "__main__":
    unittest.main()
