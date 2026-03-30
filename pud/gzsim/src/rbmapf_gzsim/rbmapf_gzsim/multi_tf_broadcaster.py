from functools import partial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class MultiTfBroadcaster(Node):
    def __init__(self):
        super().__init__('multi_tf_broadcaster')

        self.declare_parameter('namespaces', '/crazyflie_1,/crazyflie_2')
        ns_list = self.get_parameter('namespaces').get_parameter_value().string_value
        self.namespaces = [ns.strip() for ns in ns_list.split(',') if ns.strip()]

        self.latest = {}
        self.tf_broadcaster = TransformBroadcaster(self)

        for namespace in self.namespaces:
            topic = f'{namespace}/odom'
            self.create_subscription(
                Odometry,
                topic,
                partial(self.odom_callback, namespace=namespace),
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")

    def odom_callback(self, msg: Odometry, namespace: str):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest[namespace] = (
            msg.header.stamp,
            (p.x, p.y, p.z),
            (q.x, q.y, q.z, q.w)
        )
        self.publish_tf(namespace)

    def publish_tf(self, namespace: str):
        data = self.latest.get(namespace)
        if data is None:
            return

        stamp, (x, y, z), (qx, qy, qz, qw) = data
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'{namespace}/base_footprint'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MultiTfBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
