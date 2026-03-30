import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from rbmapf_gzsim.control_pointenv import argument_parser
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        self.declare_parameter('interface', 'px4')
        self.declare_parameter('risk_bound_percent', 0.5)
        self.declare_parameter('problem_index', 0)
        args = argument_parser()
        args.risk_bound_percent = (
            self.get_parameter('risk_bound_percent')
            .get_parameter_value()
            .double_value
        )
        args.problem_index = int(
            self.get_parameter('problem_index')
            .get_parameter_value()
            .integer_value
        )
        habitat = args.visual == 'True'

        if habitat:
            from rbmapf_gzsim.control_habitatenv import generate_wps
        else:
            from rbmapf_gzsim.control_pointenv import generate_wps

        self.get_logger().info(
            "Using risk_bound_percent="
            f"{args.risk_bound_percent:.2f}, problem_index={args.problem_index}"
        )
        agents_waypoints = generate_wps(args, debug=False)

        self.waypoint_publishers = []
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        interface = self.get_parameter('interface').get_parameter_value().string_value
        for agent_idx in range(args.num_agents):
            instance_idx = agent_idx + 1
            if interface != "cf":
                wp_topic = f"/{interface}_{instance_idx}/waypoints"
            else:
                wp_topic = f"/{interface}{instance_idx}/waypoints"
            waypoint_publisher = self.create_publisher(
                Float32MultiArray,
                wp_topic,
                qos_profile
            )
            agent_wps = np.array(agents_waypoints[agent_idx])
            message = Float32MultiArray(data=agent_wps.flatten().tolist())
            waypoint_publisher.publish(message)
            self.waypoint_publishers.append(waypoint_publisher)

        self.start_pub = self.create_publisher(Empty, '/start_mission', qos_profile)
        self.start_timer = self.create_timer(1.0, self.publish_start)

    def publish_start(self):
        self.start_pub.publish(Empty())
        self.get_logger().info('Mission start published')
        self.start_timer.cancel()


def main():
    rclpy.init()
    node = WaypointGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
