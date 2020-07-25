#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
from helpers.listener import BaseListener
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point
# from ads_dv_msgs.msg import VCU2AIWheelspeeds


class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        # Assign member variables
        # self.pose
        # self.prev_pose
        # self.correlations

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        # self.wss_sub = self.create_subscription(WheelSpeeds, '/can/ws', self.wss_callback, 10)

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)

        # multiple subscribers - not finished
        # cones_sub = message_filters.Subscriber(self, type, '/cones/positions')
        # odom_sub = message_filters.Subscriber(self, type, '/gazebo/odom')
        # gps_sub = message_filters.Subscriber(self, type, '/gps/data')
        # wss_sub = message_filters.Subscriber(self, type, '/wss')

        # ts = message_filters.TimeSynchronizer([points_sub, boxes_sub], 100)
        # ts.registerCallback(self.callback)
        # end multiple subscribers

    # Example callback function
    # def callback(self, msg: String):
    #     self.get_logger().info("Received: {msg.data}")

    #     if not self.count_subscribers(self.pub_topic):
    #         return

    #     self.pub.publish(String(data=msg.data))

    def cones_callback(self, msg: ConeArray):

        # Log data retrieval
        for cone in msg.cones:
            self.get_logger().info(f'Cone retrieved: Position: {cone.position}, Label: {cone.label}, Confidence: {cone.confidence}')

        # Compose ConeArray

        # Publish ConeArray to map topic
        # self.map_pub.publish(ConeArray)

    def gnss_callback(self, msg: NavSatFix()):
        # Log data retrieval
        self.get_logger().info(f'From GNSS: {msg.latitude}, {msg.longitude}')
    
    def imu_callback(self, msg: IMU()):
        # Log data retrieval
        self.get_logger().info(f'From IMU: {msg.longitudinal}, {msg.lateral}, {msg.vertical}')

    # def wss_callback(self):
        # Get WSS data
        # msg = WheelSpeeds()

        # Log data retrieval
        # self.get_logger().info('From WSS: %s' % msg)

    # def compute_pose(self):
        
        # Use GNSS, IMU, WSS to process location

        # Publish pose to position topic
        # self.pose_pub.publish(CarPos(position=Point(x=float(self.pos[0, 0]), y=float(self.pos[0, 1])), angle=float(self.pos[0, 2])))

def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)

