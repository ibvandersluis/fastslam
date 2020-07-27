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

# STEP 1: PREDICT

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

# STEP 2: UPDATE

def observation(xTrue, xd, u, RFID):

    # calc true state
    xTrue = motion_model(xTrue, u)

    # add noise to range observation
    z = np.zeros((3, 0))
    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.sqrt(dx**2 + dy**2)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Qsim[0, 0]  # add noise
            anglen = angle + np.random.randn() * Qsim[1, 1]  # add noise
            zi = np.array([dn, pi_2_pi(anglen), i]).reshape(3, 1)
            z = np.hstack((z, zi))

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1] + OFFSET_YAWRATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def update_with_observation(particles, z):
    for iz in range(len(z[0, :])):

        lmid = int(z[2, iz])

        for ip in range(N_PARTICLE):
            # new landmark
            if abs(particles[ip].lm[lmid, 0]) <= 0.01:
                particles[ip] = add_new_lm(particles[ip], z[:, iz], Q)
            # known landmark
            else:
                w = compute_weight(particles[ip], z[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], z[:, iz], Q)

    return particles

def compute_weight(particle, z, Q):
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)
    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singuler")
        return 1.0

    num = math.exp(-0.5 * dx.T @ invS @ dx)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))
    w = num / den

    return w

def compute_jacobians(particle, xf, Pf, Q):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx**2 + dy**2
    d = math.sqrt(d2)

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q

    return zp, Hv, Hf, Sf

def add_new_lm(particle, z, Q):

    r = z[0]
    b = z[1]
    lm_id = int(z[2])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s

    # covariance
    Gz = np.array([[c, -r * s],
                   [s, r * c]])

    particle.lmP[2 * lm_id:2 * lm_id + 2] = Gz @ Q @ Gz.T

    return particle

def update_KF_with_cholesky(xf, Pf, v, Q, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q

    S = (S + S.T) * 0.5
    SChol = np.linalg.cholesky(S).T
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt @ SCholInv
    W = W1 @ SCholInv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P

def update_landmark(particle, z, Q):

    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_KF_with_cholesky(xf, Pf, dz, Q, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle

# END OF CODE SNIPPET #



# Setting up the landmarks
RFID = np.array([[10.0, -2.0],
                [15.0, 10.0]])
N_LM = RFID.shape[0]

# Initialize 1 particle
N_PARTICLE = 1
particles = [Particle(N_LM) for i in range(N_PARTICLE)]

xTrue = np.zeros((STATE_SIZE, 1))
xDR = np.zeros((STATE_SIZE, 1))

print("initial weight", particles[0].w)

xTrue, z, _, ud = observation(xTrue, xDR, u, RFID)
# Initialize landmarks
particles = update_with_observation(particles, z)
print("weight after landmark initialization", particles[0].w)
particles = update_with_observation(particles, z)
print("weight after update ", particles[0].w)

particles[0].x = -10
particles = update_with_observation(particles, z)
print("weight after wrong prediction", particles[0].w)

# STEP 3: RESAMPLE



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

