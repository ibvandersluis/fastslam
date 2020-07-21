#!/usr/bin/env python3

import sys

import rclpy
from helpers.listener import BaseListener
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String


class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        self.declare_parameters('', [
            ('pub_topic', 'out', ParameterDescriptor(
                name='pub_topic',
                read_only=True,
                description='topic to send to',
                type=ParameterType.PARAMETER_STRING,
                additional_constraints='valid topic name')),
            ('sub_topic', 'in', ParameterDescriptor(
                name='sub_topic',
                read_only=True,
                description='topic to receive on',
                type=ParameterType.PARAMETER_STRING,
                additional_constraints='valid topic name')),
        ])

        self.pub_topic = self.get_parameter('pub_topic').value
        self.sub_topic = self.get_parameter('sub_topic').value

        self.pub = self.create_publisher(String, self.pub_topic, 10)
        self.sub = self.create_subscription(String, self.sub_topic, self.callback, 10)

    def callback(self, msg: String):
        self.get_logger().info(f"Received: {msg.data}")

        if not self.count_subscribers(self.pub_topic):
            return

        self.pub.publish(String(data=msg.data))


def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
