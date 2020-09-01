"""

A fresh attempt at FastSLAM

"""

#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
from helpers.listener import BaseListener
from helpers import shortcuts
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3

# FastSLAM Algorithm, as proposed by Michael Montemerlo and Sebastian Thrun
# According to their 2002 paper FastSLAM: A Factored Solution to the Simultaneous
# Localization and Mapping Problem

# Declare global variables


def fastslam():
    predict()
    update()
    resample()

def predict():
    # Odometry: xt given from motion_model(x, u)


def update():
    # w = abs(2 * pi * Q) * exp (-1/2(zt - zExp)T * Q^-1 * (zt - zExp))

def resample():




# ROS 2 Code
class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        
def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)