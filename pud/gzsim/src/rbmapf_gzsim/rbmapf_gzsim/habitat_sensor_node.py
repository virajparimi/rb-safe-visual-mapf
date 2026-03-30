import json
import yaml
import numpy as np
from dotmap import DotMap

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from rbmapf_interfaces.msg import HabitatObservations  # type: ignore

from pud.utils import set_env_seed, set_global_seed
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    safe_habitat_env_load_fn,
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
)


class HabitatSensorNode(Node):
    def __init__(self):
        super().__init__('habitat_sensor_node')

        self.declare_parameter('config_file', 'config.yaml')
        self.declare_parameter('drone_namespaces', '/crazyflie_1,/crazyflie_2')

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        ns_list = self.get_parameter('drone_namespaces').get_parameter_value().string_value
        drone_namespaces = [ns.strip() for ns in ns_list.split(',') if ns.strip()]

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)

        config.env.simulator_settings.scene_dataset = "../../" + config.env.simulator_settings.scene_dataset

        set_global_seed(config.seed)

        gym_env_wrappers = []
        gym_env_wrapper_kwargs = []
        for wrapper_name in config.wrappers:
            if wrapper_name == "GoalConditionedHabitatPointWrapper":
                gym_env_wrappers.append(GoalConditionedHabitatPointWrapper)
                gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
            elif wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
                gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
                gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
            elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
                gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
                gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

        self.eval_env = safe_habitat_env_load_fn(
            env_kwargs=config.env.toDict(),
            cost_f_args=config.cost_function.toDict(),
            cost_limit=config.agent_cost_kwargs.cost_limit,
            max_episode_steps=config.time_limit.max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,  # type: ignore
            wrapper_kwargs=gym_env_wrapper_kwargs,
            terminate_on_timeout=True,
        )
        set_env_seed(self.eval_env, config.seed + 1)

        self.bridge = CvBridge()

        self.image_publishers = {}
        for drone_ns in drone_namespaces:
            self.get_logger().info(f"Setting up sensor for drone namespace: {drone_ns}")
            state_topic = f'{drone_ns}/state'
            image_topic = f'{drone_ns}/camera/habitat_observations'
            self.image_publishers[drone_ns] = self.create_publisher(HabitatObservations, image_topic, 10)
            self.create_subscription(String, state_topic, lambda msg, ns=drone_ns: self.state_callback(ns, msg), 10)

        self.get_logger().info("Habitat Sensor Node has been started.")

    def state_callback(self, drone_ns, msg):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode state message: {e}")
            return

        self.latest_state = state
        goal_grid = self.latest_state.get('goal_grid', None)
        observation_grid = self.latest_state.get('observation_grid', None)

        if goal_grid is not None and observation_grid is not None:
            image_msg = HabitatObservations()

            goal_grid = np.array(goal_grid, dtype=np.float32)
            goal_visual = self.eval_env.get_sensor_obs_at_grid_xy(goal_grid)

            observation_grid = np.array(observation_grid, dtype=np.float32)
            observation_visual = self.eval_env.get_sensor_obs_at_grid_xy(observation_grid)

            stamp = self.get_clock().now().to_msg()
            for idx, direction in enumerate(['forward', 'right', 'backward', 'left']):
                goal_img = self.bridge.cv2_to_imgmsg(goal_visual[idx].astype(np.uint8))
                goal_img.header.stamp = stamp
                setattr(image_msg, f'goal_{direction}', goal_img)
                observation_img = self.bridge.cv2_to_imgmsg(observation_visual[idx].astype(np.uint8))
                observation_img.header.stamp = stamp
                setattr(image_msg, f'observation_{direction}', observation_img)

            self.image_publishers[drone_ns].publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HabitatSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
