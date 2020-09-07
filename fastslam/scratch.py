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
    # For all particles:
    # Odometry: xt given from motion_model(x, u)
    # Make measurement prediction
    # Calculate H (jacobian)
    # Calculate Q (measurement covariance)
    # Calculate w (likelihood of correspondence)

    # w[k] = max(wj)
    # c_hat = argmax(w)
    # N checked
    # For all landmarks
        # If j new feature,
            # Initialise mean
            # Initialise covariance
            # Initialise counter (haven't done this yet)
        # Else if it is an observed feature
            # Calculate Kalman gain
            # Update mean
            # Update covariance
            # Increment counter
        # Else
            # Should landmark have been seen?
            # Yes -> decrement counter
            # No -> do nothing
            # If counter < 0, remove
    # Resample particles M with probability derived from weight
    # Return new particles


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