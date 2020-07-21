#!/usr/bin/env python3

import sys

import message_filters
import rclpy
from helpers.listener import BaseListener
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from msgs.msg import String


class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        # publish to /rosout
        self.pub = self.create_publisher(String, '/rosout', 10)
        self.create_subscription(String, '/wheelspeed', self.callback, 10)

        # multiple subscribers - not finished
        cones_sub = message_filters.Subscriber(self, type, '/cones/positions')
        odom_sub = message_filters.Subscriber(self, type, '/gazebo/odom')
        gps_sub = message_filters.Subscriber(self, type, '/gps/data')
        wss_sub = message_filters.Subscriber(self, type, '/wss')

        ts = message_filters.TimeSynchronizer([points_sub, boxes_sub], 100)
        ts.registerCallback(self.callback)
        # end multiple subscribers

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
