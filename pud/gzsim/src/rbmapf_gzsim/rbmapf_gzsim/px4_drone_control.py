import yaml
import torch
import argparse
import numpy as np
from dotmap import DotMap

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from std_msgs.msg import Empty, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy,
    QoSReliabilityPolicy, QoSDurabilityPolicy,
)
from px4_msgs.msg import (  # type: ignore
    VehicleCommand, VehicleStatus,
    TrajectorySetpoint, OffboardControlMode, VehicleOdometry,
)

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag


def now_us(clock):
    return int(clock.now().nanoseconds / 1000)


def enu_to_ned(enu):
    return np.array([enu[1], enu[0]], dtype=float)


def ned_to_enu(ned):
    return np.array([ned[1], ned[0]], dtype=float)


class DroneController(Node):
    def __init__(self, drone_ns, drone_id, num_drones, files, waypoint_follow=False, gz_version='harmonic'):
        super().__init__(f"drone_controller_{drone_id}")
        self.start_flag = False
        self.drone_ns = drone_ns
        self.drone_id = drone_id
        self.gz_version = gz_version
        self.num_drones = num_drones
        self.waypoint_follow = waypoint_follow
        config_file, ckpt_file, walls_file = files

        self.walls = np.load(walls_file)
        self.normalize_factor = np.array([self.walls.shape[0], self.walls.shape[1]])
        self.origin_offset = -self.normalize_factor / 2.0

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_agent(config, ckpt_file)

        self._init_qos()
        self._init_publishers()
        self._init_subscribers()
        self._init_visualizers()

        self.counter = 0
        self.arm_timer = self.create_timer(0.1, self.arm_timer_callback)
        callback_fn = self.waypoint_follower_callback if self.waypoint_follow else self.command_callback
        self.cmd_timer = self.create_timer(0.02, callback_fn)

        self.altitude = 2.0
        self.altitude += (self.drone_id - 2) * 0.5
        self.failsafe = False
        self.flight_check = False
        self.current_state = "IDLE"
        self.offboard_mode = False
        self.distance_threshold = 0.5
        self.local_position = np.zeros(2)
        self.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.other_agent_homes = {}
        self.other_agent_positions = {}

    def _init_agent(self, config, ckpt_file):
        self.agent = DRLDDPGLag(
            4, 2, 1,
            CriticCls=GoalConditionedCritic,
            device=torch.device(config.device),
            **config.agent,
        )
        self.agent.load_state_dict(
            torch.load(ckpt_file, map_location=torch.device(config.device))
        )
        self.agent.to(torch.device(config.device))
        self.agent.eval()

    def _init_qos(self):
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

    def _init_subscribers(self):
        self.waypoint_subscriber = self.create_subscription(
            Float32MultiArray,
            f"{self.drone_ns}/waypoints",
            self.waypoints_callback,
            self.qos_profile
        )
        vehicle_status_topic_suffix = "_v1" if self.gz_version == 'classic' else ''
        self.status_subscriber = self.create_subscription(
            VehicleStatus,
            f"{self.drone_ns}/fmu/out/vehicle_status{vehicle_status_topic_suffix}",
            self.vehicle_status_callback,
            self.qos_profile,
        )
        self.create_subscription(
            Empty,
            "/start_mission",
            self.start_callback,
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.odom_subscriber = self.create_subscription(
            VehicleOdometry,
            f"{self.drone_ns}/fmu/out/vehicle_odometry",
            self.local_position_callback,
            self.qos_profile,
        )
        for i in range(1, self.num_drones + 1):
            drone_ns = f"/px4_{i}"
            if drone_ns != self.drone_ns:
                self.create_subscription(
                    VehicleOdometry,
                    f"{drone_ns}/fmu/out/vehicle_odometry",
                    lambda msg, ns=drone_ns: self.other_position_callback(ns, msg),
                    self.qos_profile,
                )
                self.create_subscription(
                    Float32MultiArray,
                    f"{drone_ns}/waypoints",
                    lambda msg, ns=drone_ns: self.other_waypoints_callback(ns, msg),
                    self.qos_profile,
                )

    def _init_publishers(self):
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            f"{self.drone_ns}/fmu/in/offboard_control_mode",
            self.qos_profile,
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            f"{self.drone_ns}/fmu/in/vehicle_command",
            10,
        )
        self.publisher_trajectory_setpoint = self.create_publisher(
            TrajectorySetpoint,
            f"{self.drone_ns}/fmu/in/trajectory_setpoint",
            self.qos_profile,
        )

    def _init_visualizers(self):
        self.trajectory = Path()
        self.trajectory.header.frame_id = "world"
        self.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8),
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8),
            ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8),
            ColorRGBA(r=0.5, g=0.0, b=1.0, a=0.8),
        ]

        self.path_publisher = self.create_publisher(
            Path,
            f"{self.drone_ns}/trajectory",
            10
        )
        self.waypoint_marker_publisher = self.create_publisher(
            MarkerArray,
            f"{self.drone_ns}/waypoint_markers",
            10
        )
        self.wall_publisher = self.create_publisher(MarkerArray, 'wall_markers', 10)
        self.publish_wall_markers()

    def _marker_color(self):
        return self.colors[(self.drone_id - 2) % len(self.colors)]

    def publish_markers(self):
        markers = MarkerArray()
        marker_color = self._marker_color()
        for idx, (xe, yn, _) in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'waypoints'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xe)
            marker.pose.position.y = float(yn)
            marker.pose.position.z = self.altitude
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color = marker_color
            marker.lifetime = Duration(sec=0)
            markers.markers.append(marker)  # type: ignore

        start_text = Marker()
        start_text.header.frame_id = 'world'
        start_text.header.stamp = self.get_clock().now().to_msg()
        start_text.ns = 'start'
        start_text.id = 0
        start_text.type = Marker.TEXT_VIEW_FACING
        start_text.action = Marker.ADD
        start_text.pose.position.x = float(self.waypoints[0][0])
        start_text.pose.position.y = float(self.waypoints[0][1])
        start_text.pose.position.z = self.altitude + 0.5
        start_text.text = f"START {self.drone_id - 1}"
        start_text.scale.z = 0.4
        start_text.color = marker_color
        start_text.lifetime = Duration(sec=0)
        markers.markers.append(start_text)  # type: ignore

        goal_text = Marker()
        goal_text.header.frame_id = 'world'
        goal_text.header.stamp = self.get_clock().now().to_msg()
        goal_text.ns = 'goal'
        goal_text.id = 0
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.pose.position.x = float(self.waypoints[-1][0])
        goal_text.pose.position.y = float(self.waypoints[-1][1])
        goal_text.pose.position.z = self.altitude + 0.5
        goal_text.text = f"GOAL {self.drone_id - 1}"
        goal_text.scale.z = 0.4
        goal_text.color = marker_color
        goal_text.lifetime = Duration(sec=0)
        markers.markers.append(goal_text)  # type: ignore

        self.waypoint_marker_publisher.publish(markers)

    def publish_wall_markers(self, height=5.0, resolution=1.0):
        marker_array = MarkerArray()

        idx = 0
        for i, j in zip(*np.where(self.walls == 1)):
            x = self.origin_offset[0] + (j + 0.5)*resolution
            y = self.origin_offset[1] + (i + 0.5)*resolution

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'walls'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(height / 2.0)
            marker.scale.x = marker.scale.y = resolution
            marker.scale.z = height
            marker.color = ColorRGBA(r=0.3, g=0.3, b=0.3, a=1.0)
            marker.lifetime = Duration(sec=0)
            marker_array.markers.append(marker)  # type: ignore
            idx += 1

        self.wall_publisher.publish(marker_array)

    def waypoints_callback(self, msg: Float32MultiArray):
        # Waypoints are in global ENU frame with first entry as home, last entry as goal
        self.current_wp_index = 1
        self.waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        self.publish_markers()

        self.home = self.waypoints[0][:2].copy()
        # State needs to be in the coordinate frame that the GCRL policy was trained with
        # i.e origin is bottom left corner of the walls matrix not the origin of simulator!
        self.state = self.home.copy() - self.origin_offset
        self.next_location = np.zeros(2)
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints")

    def other_waypoints_callback(self, drone_ns, msg):
        other_waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        other_home = other_waypoints[0][:2].copy()
        self.other_agent_homes[drone_ns] = other_home

    def vehicle_status_callback(self, msg):
        if msg.nav_state != self.nav_state:
            self.get_logger().info(f"{self.drone_ns} NAV_STATUS: {msg.nav_state}")

        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"{self.drone_ns} ARM STATUS: {msg.arming_state}")

        if msg.failsafe != self.failsafe:
            self.get_logger().info(f"{self.drone_ns} FAILSAFE: {msg.failsafe}")

        if msg.pre_flight_checks_pass != self.flight_check:
            print(msg)
            self.get_logger().info(
                f"{self.drone_ns} FlightCheck: {msg.pre_flight_checks_pass}"
            )

        self.failsafe = msg.failsafe
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.flight_check = msg.pre_flight_checks_pass

    def start_callback(self, msg):
        self.start_flag = True
        self.get_logger().info(f"Drone {self.drone_id} received start signal.")

    def local_position_callback(self, msg):
        # if self.gz_version == 'harmonic':
        #     # Local position is in NED frame
        #     self.local_position[0] = msg.position[0]
        #     self.local_position[1] = msg.position[1]
        # else:
        #     # Global position coming from VICON is in ENU frame
        #     # Extract the local position from the global location
        #     self.local_position[0] = msg.pose.position.x - self.home[0]
        #     self.local_position[1] = msg.pose.position.y - self.home[1]

        #     # Convert to NED
        #     self.local_position = enu_to_ned(self.local_position)

        # TODO: Replace this with the correct version above after debugging
        # Local position is in NED frame
        self.local_position[0] = msg.position[0]
        self.local_position[1] = msg.position[1]

    def other_position_callback(self, drone_ns, msg):
        # Other drone position is in NED frame
        other_position = np.array([msg.position[0], msg.position[1]], dtype=float)
        other_position_enu = ned_to_enu(other_position)
        if drone_ns in self.other_agent_homes:
            self.other_agent_positions[drone_ns] = self.other_agent_homes[drone_ns] + other_position_enu

    def arm_timer_callback(self):
        if self.current_state == "IDLE":
            if self.flight_check:
                self.current_state = "ARMING"
                self.get_logger().info(f"Arming to {self.drone_ns}")
        elif self.current_state == "ARMING":
            if not self.flight_check:
                self.current_state = "IDLE"
                self.get_logger().info(f"Arming, Flight Check Failed to {self.drone_ns}")
            elif self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self.counter > 10:
                self.current_state = "TAKEOFF"
                self.get_logger().info(f"Arming, Takeoff to {self.drone_ns}")
            self.arm()
        elif self.current_state == "TAKEOFF":
            if not self.flight_check:
                self.current_state = "IDLE"
                self.get_logger().info(f"Takeoff, Flight Check Failed to {self.drone_ns}")
            elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                self.current_state = "LOITER"
                self.get_logger().info(f"Takeoff, Loiter to {self.drone_ns}")
            self.arm()
            self.takeoff()
        elif self.current_state == "LOITER":
            if not self.flight_check:
                self.current_state = "IDLE"
                self.get_logger().info(f"Loiter, Flight Check Failed to {self.drone_ns}")
            elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                self.current_state = "OFFBOARD"
                self.get_logger().info(f"Loiter, Offboard to {self.drone_ns}")
            self.arm()
        elif self.current_state == "OFFBOARD":
            if not self.flight_check or self.arm_state != VehicleStatus.ARMING_STATE_ARMED or self.failsafe:
                self.current_state = "IDLE"
                self.get_logger().info(f"Offboard, Flight Check Failed to {self.drone_ns}")
            self.offboard()

        self.counter += 1

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
        )
        self.get_logger().info(f"Arm command sent to {self.drone_ns}")

    def takeoff(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1=1.0, param7=5.0
        )
        self.get_logger().info(f"Takeoff command sent to {self.drone_ns}")

    def offboard(self):
        self.counter = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.offboard_mode = True

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.command = command
        msg.target_system = self.drone_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = now_us(self.get_clock())
        self.vehicle_command_publisher.publish(msg)

    def send_offboard(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us(self.get_clock())
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

    def send_debug_trajectory(self, position):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = self.altitude
        self.trajectory.poses.append(pose)  # type: ignore
        self.path_publisher.publish(self.trajectory)

    def send_trajectory(self, position):
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = now_us(self.get_clock())
        position_ned = enu_to_ned(position)
        trajectory_msg.position[0] = position_ned[0]
        trajectory_msg.position[1] = position_ned[1]
        trajectory_msg.position[2] = -self.altitude
        self.publisher_trajectory_setpoint.publish(trajectory_msg)

    def waypoint_follower_callback(self):
        if self.offboard_mode and self.start_flag:
            self.send_offboard()
            if hasattr(self, "waypoints") and self.current_wp_index < len(self.waypoints):

                # Waypoints are in global ENU frame
                target = self.waypoints[self.current_wp_index][:2].copy()
                # Extact the waypoint position in local ENU frame
                target_relative = target - self.home
                self.send_trajectory(target_relative)

                local_enu_position = ned_to_enu(self.local_position)
                if np.linalg.norm(local_enu_position - target_relative) < self.distance_threshold:
                    self.get_logger().info(
                        f"Waypoint {self.current_wp_index} reached for {self.drone_ns}"
                    )
                    self.current_wp_index += 1

    def command_callback(self):
        # Uses the low-level trained GC-RL policy
        if self.offboard_mode and self.start_flag:
            if hasattr(self, "waypoints"):
                if self.current_wp_index < len(self.waypoints):
                    self.send_offboard()

                    local_enu_position = ned_to_enu(self.local_position)
                    global_enu_position = self.home + local_enu_position
                    target = self.waypoints[self.current_wp_index][:2].copy()
                    self.send_debug_trajectory(global_enu_position)

                    # Updates the high-level waypoints towards the main goal
                    if np.linalg.norm(global_enu_position - target) < self.distance_threshold:
                        self.get_logger().info(
                            f"Waypoint {self.current_wp_index} reached for {self.drone_ns}"
                        )
                        self.current_wp_index += 1
                        if self.current_wp_index >= len(self.waypoints):
                            self.get_logger().info(f"Mission completed for {self.drone_ns}")
                            return

                    # Updates the low-level waypoints towards the next location
                    if np.linalg.norm(local_enu_position - self.next_location) < self.distance_threshold:
                        # Input to the agent is normalized global ENU positions
                        observation = self.state / self.normalize_factor
                        goal = (
                            self.waypoints[self.current_wp_index][:2].copy() - self.origin_offset
                        ) / self.normalize_factor
                        state = {
                            "observation": observation,
                            "goal": goal,
                        }
                        action = self.agent.select_action(state)
                        # Output of applying the action is in global ENU frame
                        observation = self.step(action)
                        observation += self.origin_offset

                        self.next_location = observation - self.home

                    for drone_ns, other_position in self.other_agent_positions.items():
                        self_id = int(self.drone_ns.split("_")[-1])
                        other_id = int(drone_ns.split("_")[-1])
                        distance = np.linalg.norm(self.next_location + self.home - other_position)
                        if distance < self.distance_threshold * 2 and other_id < self_id:
                            self.get_logger().info(
                                f"Drone {drone_ns} is too close to {self.drone_ns}, waiting at current location"
                            )
                            self.send_trajectory(ned_to_enu(self.local_position))
                            return

                    self.send_trajectory(self.next_location)

                else:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int64)
        if i == self.walls.shape[0]:
            i -= 1
        if j == self.walls.shape[1]:
            j -= 1
        return (i, j)

    def is_blocked(self, state):
        if not (np.all(state >= np.zeros(2)) and np.all(state <= self.normalize_factor)):
            return True
        (i, j) = self.discretize_state(state)
        return (self.walls[i, j] == 1)

    def step(self, action):
        action = np.clip(action, -1. * np.ones(2), np.ones(2))
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self.is_blocked(new_state):
                    self.state = new_state

        return self.state.copy()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Multi-drone controller launch')
    parser.add_argument(
        '--drone_ids',
        type=str,
        default='2,3',
        help='Comma-separated list of drone system IDs'
    )
    parser.add_argument(
        '--drone_namespaces',
        type=str,
        default='/px4_1,/px4_2',
        help='Comma-separated list of drone ROS namespaces'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--ckpt_file',
        type=str,
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--walls_file',
        type=str,
        help='Path to the walls file'
    )
    parser.add_argument(
        '--gz_version',
        type=str,
        default="harmonic",
        choices=['harmonic', 'classic'],
        help='Gazebo version to use'
    )
    args, _ = parser.parse_known_args()

    drone_ids = [int(x) for x in args.drone_ids.split(',')]
    drone_namespaces = args.drone_namespaces.split(',')
    drone_nodes = [
        DroneController(ns, nid, len(drone_ids), (args.config_file, args.ckpt_file, args.walls_file),
                        gz_version=args.gz_version)
        for _, (ns, nid) in enumerate(zip(drone_namespaces, drone_ids))
    ]

    try:
        executor = MultiThreadedExecutor()
        for node in drone_nodes:
            executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in drone_nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
