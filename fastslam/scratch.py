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

def point_angle_line(x, y, theta):
    """
    Plot a line from slope and intercept

    :param rads: The angle of the line in radians
    :param x: The x value of the point
    :point y: The y value of the point
    :return: Nothing
    """
    slope = np.tan(theta)
    intercept = y - slope * x
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def observation_model(particle):
    """
    Compute predictions for expected observation of landmarks based on expected pose
    of vehicle at next time step, computed by motion model

    :param particle: A particle
    :return: The particle with landmark predictions added to landmark array
    """

    # Get expected observations for landmarks
    for i in range(len(particle.mu[:, 0])): # For each landmark in the particle
        mu = np.array(particle.mu[i, 0:2]).reshape(2, 1) # Let mu be the x, y values of the particle
        dxy = mu - particle.x[0:2, 0] # Get relative displacement between landmark and pose
        d_sq = np.sum(dxy**2) # Pythagorean theorem
        d = np.sqrt(d_sq) # Get relative distance from vehicle
        theta = pi_2_pi(np.arctan2(dxy[1, 0], dxy[0, 0]) - particle.x[2, 0]) # Get relative angle of observation

        particle.mu[i, 2] = d # Assign expected distance
        particle.mu[i, 3] = theta # Assign expected angle of observation

    return particle

def law_of_cos(a, b, theta):
    """
    Returns the length of the side of a triangle opposite a corner,
    given that corner's angle and the lengths of the adjacent sides.

    :param a: The length of the first side of the triangle
    :param b: The length of the second side of the triangle
    :param theta: The measure of the angle in radians between the two sides
    :return: c, the length of the third side
    """
    c_sq = a**2 + b**2 - 2*a*b * np.cos(theta)
    c = np.sqrt(c_sq)

    return c

# Declare global variables


def fastslam():

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

    predict()
    update()
    resample()

def predict():
    # For all particles:
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