#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
import math
from copy import deepcopy
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point
# from ads_dv_msgs.msg import VCU2AIWheelspeeds

shortcuts.hint()

# --- CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)])**2
R = np.diag([1.0, np.deg2rad(20.0)])**2

#  Simulation parameter
Qsim = np.diag([0.3, np.deg2rad(2.0)])**2
Rsim = np.diag([0.5, np.deg2rad(10.0)])**2
OFFSET_YAWRATE_NOISE = 0.01

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

class Particle:
    def __init__(self, N_LM):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((N_LM * LM_SIZE, LM_SIZE))

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])
    x = F @ x + B @ u

    x[2, 0] = pi_2_pi(x[2, 0])
    return x

def predict_particles(particles, u):
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        ud = u + (np.random.randn(1, 2) @ R).T  # add noise
        px = motion_model(px, ud)
        particles[i].x = px[0, 0]
        particles[i].y = px[1, 0]
        particles[i].yaw = px[2, 0]

    return particles

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# END OF SNIPPET

N_LM = 0
particles = [Particle(N_LM) for i in range(N_PARTICLE)]
time= 0.0
v = 1.0  # [m/s]
yawrate = 0.1  # [rad/s]
u = np.array([v, yawrate]).reshape(2, 1)
history = []
while SIM_TIME >= time:
    time += DT
    particles = predict_particles(particles, u)
    history.append(deepcopy(particles))

# --- END CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

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
            self.get_logger().info(str(cone))

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

