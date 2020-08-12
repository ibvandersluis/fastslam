"""

FastSLAM 1.0 for use with ROS 2

Author: Isaac Vander Sluis
Starter code: Atsushi Sakai

"""

#!/usr/bin/env python3

import sys

import message_filters
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import deepcopy
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates
# from ads_dv_msgs.msg import VCU2AIWheelspeeds

shortcuts.hint()

def observation_model(particles, x, u, z):
    """
    Compute predictions for relative location of landmarks

    :param particles: An array of particles
    :param x: The state vector
    :param u: The input vector: velocity and yaw
    :param z: The observations
    :return:
    """

    pnum = 0
    for particle in particles:
        pnum += 1
        for i in range(len(particle.lm[0, :])):
            xf = np.array(particle.lm[i, :]).reshape(3, 1)
            dx = xf[0, 0] - particle.x
            dy = xf[1, 0] - particle.y
            d2 = dx ** 2 + dy ** 2
            d = math.sqrt(d2)

            zp = np.array([d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)
            
            dz = z[i, 0:2].reshape(2, 1) - zp
            dz[1, 0] = pi_2_pi(dz[1, 0])




# --- CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

# Code mostly by Atsushi Sakai
# Comments mostly my own

# Python Robotics
#   https://pythonrobotics.readthedocs.io/en/latest/
# Python Robotics: Localization
#   https://pythonrobotics.readthedocs.io/en/latest/modules/localization.html
# Python Robotics: SLAM
#   https://pythonrobotics.readthedocs.io/en/latest/modules/slam.html
# Atsushi Sakai on GitHub
#   https://github.com/AtsushiSakai
# FastSLAM 1.0 code (starter code used here)
#   https://github.com/AtsushiSakai/PythonRobotics/blob/master/SLAM/FastSLAM1/fast_slam1.py

# STEP 1: PREDICT

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2 # Covariance matrix of process noise
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2 # Covariance matrix of observation noise at time t

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.2  # time tick [s]
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 3  # LM state size [x, y, probability]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling
PARTICLE_ITERATION = 0 # n for the nth particle production

class Particle:

    def __init__(self, n_landmark):
        """
        Construct a new particle

        :param n_landmark: The landmark number
        :return: Returns nothing
        """
        global PARTICLE_ITERATION
        PARTICLE_ITERATION += 1
        print('Creating particle #' + str(PARTICLE_ITERATION))
        
        self.w = 1.0 / N_PARTICLE # Particle weight
        self.x = 0.0 # X pos
        self.y = 0.0 # Y pos
        self.yaw = 0.0 # Orientation
        # Landmark x-y positions
        self.lm = np.zeros((n_landmark, LM_SIZE))
        # Landmark position covariance
        self.lmP = np.zeros((n_landmark * (LM_SIZE - 1), LM_SIZE - 1))

def fast_slam1(particles, u, z):
    """
    Updates beliefs about position and landmarks using FastSLAM 1.0

    :param particles:
    :param u: The controls (velocity and orientation)
    :param z: The observation
    :return: Returns updated particles (position and landmarks)
    """
    print('RUNNING SLAM')

    # Step 1: predict
    particles = predict_particles(particles, u)

    # Step 2: update
    particles = update_with_observation(particles, z)

    # Step 3: resample
    particles = resampling(particles)

    return particles

def calc_final_state(particles):
    """
    Calculates the final state vector

    :param particles: An array of particles
    :return: xEst, the state vector
    """
    print('CALCULATING FINAL STATE')
    xEst = np.zeros((STATE_SIZE, 1)) # Empty state vector for: x, y, yaw

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    #  print(xEst)

    return xEst

def motion_model(x, u):
    """
    Compute predictions for a particle

    :param x: The state vector [x, y, yaw]
    :param u: The input vector [Vt, Wt]
    :return: Returns new state vector x
    """
    
    # A 3x3 matrix with one's passing through the diagonal
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # A 3x2 matrix
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = F @ x + B @ u # Formula: X = FX + BU

    x[2, 0] = pi_2_pi(x[2, 0]) # Ensure Theta is under pi radians

    return x

def predict_particles(particles, u):
    """
    Predict x, y, yaw values for new particles

    :param particles: An array of particles
    :param u: An input vector [Vt, Wt] where Vt = velocity and Wt = 
    :return: Returns predictions as particles
    """
    print('PREDICTING PARTICLES')

    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1)) # Creates 3x1 matrix of zeros for x, y, yaw
        px[0, 0] = particles[i].x # Populates top place in matrix with current particle x position
        px[1, 0] = particles[i].y # Populates mid place in matrix with current particle y position
        px[2, 0] = particles[i].yaw # Populates bot place in matrix with current particle yaw value
        ud = u + (np.random.randn(1, 2) @ R).T  # add noise
        px = motion_model(px, ud) # Compute predictions using motion model
        particles[i].x = px[0, 0] # Replace particle x pos with predicted value
        particles[i].y = px[1, 0] # Replace particle y pos with predicted value
        particles[i].yaw = px[2, 0] # Replace particle yaw with predicted value

    return particles

def pi_2_pi(angle):
    """
    Ensure the angle is under +/- PI radians

    :param angle: Angle in radians
    :return: Returns the angle after ensuring it is under +/- PI radians
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

# STEP 2: UPDATE

def observation(xTrue, xd, u, data):
    """
    Record an observation

    :param xTrue: The true state
    :param xd: The state expectation
    :param u: Velocity and Yaw
    :param data: The landmarks seen by the camera
    :return:
        xTrue - the true state
        z - the observation
        xd - sate expectation
        ud - Input with noise
    """
    print('MAKING OBSERVATION')

    # calc true state
    xTrue = motion_model(xTrue, u)

    # add noise to range observation
    z = np.zeros((3, 0))
    # For each landmark
    for i in range(len(data[:, 0])):
        # Get true distance d between pose and landmark
        dx = data[i, 0] # dx = x
        dy = data[i, 1] # dy = y
        d = math.hypot(dx, dy) # Distance
        angle = pi_2_pi(math.atan2(dy, dx)) # Angle
        print('Observation angle: ' + str(pi_2_pi(angle)))
        zi = np.array([d, pi_2_pi(angle), i]).reshape(3, 1) # The predicted measurement
        z = np.hstack((z, zi)) # add prediction to stack of observations

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def update_with_observation(particles, z):
    """
    Update particles using an observation

    :param particles: An array of particles
    :param z: An observation (array of landmarks, each [dist, theta, id])
    :return: Returns updated particles
    """
    print('UPDATING WITH OBSERVATION')

    # For each landmark in the observation
    for iz in range(len(z[0, :])):

        landmark_id = int(z[2, iz]) # Get landmark id

        for ip in range(N_PARTICLE):
            # Add new landmark if likelihood is less than 1%
            # if particles[ip].lm[landmark_id, 2]) <= 0.01:
            if abs(particles[ip].lm[landmark_id, 0]) <= 0.01:
                particles[ip] = add_new_landmark(particles[ip], z[:, iz], Q)
            # Else update the known landmark
            else:
                w = compute_weight(particles[ip], z[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], z[:, iz], Q)

    return particles

def compute_weight(particle, z, Q_cov):
    """
    Compute weight of particles

    :param particle: A particle
    :param z: An observation
    :param Q_cov: The measurement covariance
    :return: Returns particle weight
    """
    lm_id = int(z[2]) # Get landmark id from z
    xf = np.array(particle.lm[lm_id, :]).reshape(3, 1) # The state of a landmark from a particle
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2]) # ??
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singular")
        return 1.0

    num = math.exp(-0.5 * dx.T @ invS @ dx)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))

    w = num / den

    return w

def compute_jacobians(particle, xf, Pf, Q_cov):
    """
    Computes Jacobian matrices

    :param particle: A particle
    :param xf:
    :param Pf:
    :param Q_cov: A covariance matrix of process noise
    :return:
        zp - 
        Hv -
        Hf - 
        Sf - 
    """

    # Compute distance
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    zp = np.array([d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf


def add_new_landmark(particle, z, Q_cov):
    """
    Adds a new landmark to a particle

    :param particle: A particle
    :param z: An observation
    :param Q_cov: A covariance matrix of process noise
    :return: A particle
    """

    r = z[0]
    b = z[1]
    lm_id = int(z[2])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    # particle.lm[lm_id, 0] = particle.x + r * c
    # particle.lm[lm_id, 1] = particle.y + r * s
    np.append(particle.lm, [particle.x + r * c, particle.y + r * s, 1.0]) # Add new lm to array
    print(particle.lm)

    # covariance
    dx = r * c
    dy = r * s
    d2 = dx**2 + dy**2
    d = math.sqrt(d2) # Get distance
    Gz = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])
    particle.lmP[2 * lm_id:2 * lm_id + 2] = np.linalg.inv(Gz) @ Q_cov @ np.linalg.inv(Gz.T)

    return particle

def update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf):
    """
    Update Kalman filter

    :param xf:
    :param Pf:
    :param v: The velocity
    :param Q_cov: A covariance matrix of process noise
    :param Hf:
    :return:
        x - 
        P - 
    """
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q_cov

    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P

def update_landmark(particle, z, Q_cov):
    """
    Update a landmark

    :param particle: A particle
    :param z: An observation
    :param Q_cov: A covariance matrix of process noise
    :return: A particle
    """
    print('UPDATING LANDMARK')

    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(3, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle

# STEP 3: RESAMPLE

def normalize_weight(particles):
    """
    Applies Gaussian distribution to particle weights

    :param particles: An array of particles
    :return: An array of particles with reassigned weights
    """
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def resampling(particles):
    """
    Low-variance resampling

    :param particles: An array of particles
    :return: An array of particles
    """
    print('RESAMPLING')

    # Normalize weights
    particles = normalize_weight(particles)

    # Get particle weights
    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(n_eff)

    if n_eff < NTH:  # resampling
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)

        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].yaw = tmp_particles[inds[i]].yaw
            particles[i].lm = tmp_particles[inds[i]].lm[:, :]
            particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / N_PARTICLE

    return particles

# --- END CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

# ROS 2 Code
class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        # State variables
        self.x = None
        self.y = None
        self.v = 0.0 # Velocity, m/s
        self.yaw = 0.0 # Yaw rate, rad/s

        self.timer_last = self.get_clock().now().nanoseconds
        self.capture = [] # For cone data from snapsot of camera
        self.n_landmark = 0 # Number of initial landmdarks


        # State Vector [x y yaw]
        self.xEst = np.zeros((STATE_SIZE, 1))  # SLAM estimation
        self.xTrue = np.zeros((STATE_SIZE, 1))  # True state
        self.xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

        # History
        self.hxEst = self.xEst
        self.hxTrue = self.xTrue
        self.hxDR = self.xTrue

        # Generate initial particles
        self.particles = [Particle(self.n_landmark) for _ in range(N_PARTICLE)]


        self.u = np.array([self.v, self.yaw]).reshape(2, 1)
        self.ud = None
        self.z = None
        self.count = 0
        self.debug = 0



        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/gazebo/cmd_vel', self.control_callback, 10)
        # self.wss_sub = self.create_subscription(WheelSpeeds, '/can/ws', self.wss_callback, 10)

        # gets links (all objects) from gazebo
        self.link_sub = self.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_callback, 10)

        # ros2 topic pub /gazebo/cmd_vel geometry_msgs/Twist '{linear: {x: 1.0}, angular: {z: 0.1}}' 

        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)
        self.cmd_pub = self.create_publisher(Twist, '/gazebo/cmd_vel', 10)

        # multiple subscribers - not finished
        # cones_sub = message_filters.Subscriber(self, type, '/cones/positions')
        # odom_sub = message_filters.Subscriber(self, type, '/gazebo/odom')
        # gps_sub = message_filters.Subscriber(self, type, '/gps/data')
        # wss_sub = message_filters.Subscriber(self, type, '/wss')

        # ts = message_filters.TimeSynchronizer([points_sub, boxes_sub], 100)
        # ts.registerCallback(self.callback)
        # end multiple subscribers

    def cones_callback(self, msg: ConeArray):
        # Place x y positions of cones into self.capture
        self.capture = np.array([[cone.x, cone.y] for cone in msg.cones])
        print(self.capture)

        # Set time
        global DT
        DT = (self.get_clock().now().nanoseconds - self.timer_last)
        DT /= 1000000000 # Nanoseconds to seconds
        self.timer_last = self.get_clock().now().nanoseconds # Set timer_last as current nanoseconds
        print('DT -- ' + str(DT) + 's')

        # Get observation
        self.xTrue, self.z, self.xDR, self.ud = observation(self.xTrue, self.xDR, self.u, self.capture)

        print('xTrue Angle: ' + str(self.xTrue[2, 0]))

        # Run SLAM
        self.particles = fast_slam1(self.particles, self.ud, self.z)

        # Increment counter
        self.count += 1
        # Dump particle lm arrays to text file
        if (self.count >= 10):
            self.debug += 1
            file = 'debug' + str(self.debug) + '.txt'
            f = open(file, 'w')
            pnum = 0
            for particle in self.particles:
                pnum += 1
                f.write('--- Particle #' + str(pnum) + ':\n')
                for i in range(len(particle.lm[:, 0])):
                    f.write('lm #' + str(i + 1) + ' -- x: ' + str(particle.lm[i, 0])
                            + ', y: ' + str(particle.lm[i, 1]) + '\n')
            f.close()
            self.count = 0

        # Get state estimation
        self.xEst = calc_final_state(self.particles)
        self.x = self.xEst[0, 0]
        self.y = self.xEst[1, 0]

        # Boundary check
        self.x_state = self.xEst[0: STATE_SIZE]

        # Store data history
        self.hxEst = np.hstack((self.hxEst, self.x_state))
        self.hxDR = np.hstack((self.hxDR, self.xDR))
        self.hxTrue = np.hstack((self.hxTrue, self.xTrue))

        # Plot graph
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        # Plot landmarks as black stars relative to xEst
        # plt.plot(self.capture[:, 0] + self.xEst[0, 0], self.capture[:, 1] + self.xEst[1, 0], "*k")
        
        # Convert z observations to absolute positions and plot
        for i in range(len(self.z[0, :])):
            x = self.xEst[0, 0] # X pos
            y = self.xEst[1, 0] # Y pos
            yaw = self.xEst[2, 0] # Orientation
            d = self.z[0, i] # Distance from vehicle
            theta = self.z[1, i] # Angle of observation
            plt.plot(x + d * math.cos(pi_2_pi(theta + yaw)), y + d * math.sin(pi_2_pi(theta + yaw)), "*k")

        for i in range(N_PARTICLE):
            # Plot location estimates as red dots
            plt.plot(self.particles[i].x, self.particles[i].y, ".r")
            # Plot landmark estimates as blue X's
            plt.plot(self.particles[i].lm[:, 0], self.particles[i].lm[:, 1], "xb")

        plt.plot(self.hxTrue[0, :], self.hxTrue[1, :], "-b") # Plot xTrue with solid blue line
        plt.plot(self.hxDR[0, :], self.hxDR[1, :], "-k") # Plot dead reckoning with solid black line
        plt.plot(self.hxEst[0, :], self.hxEst[1, :], "-r") # Plot xEst with solid red line
        plt.plot(self.xEst[0], self.xEst[1], "xk") # Plot current xEst as black x
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    def control_callback(self, msg: Twist):
        str(msg) # For some reason this is needed to access msg.linear.x
        self.v = msg.linear.x
        self.yaw = msg.angular.z
        self.u = np.array([self.v, self.yaw]).reshape(2, 1)

        self.get_logger().info(f'Command confirmed: {msg.linear.x} m/s turning at {msg.angular.z} rad/s')

    def gnss_callback(self, msg: NavSatFix()):
        # Log data retrieval
        self.get_logger().info(f'From GNSS: {msg.latitude}, {msg.longitude}')
    
    def imu_callback(self, msg: IMU()):
        # Log data retrieval
        self.get_logger().info(f'From IMU: {msg.longitudinal}, {msg.lateral}, {msg.vertical}')

    def link_states_callback(self, links_msg: LinkStates):
        cones = []
        for name, pose in zip(links_msg.name, links_msg.pose):
            if 'blue_cone' in name:
                label = Label.BLUE_CONE
            elif 'yellow_cone' in name:
                label = Label.YELLOW_CONE
            elif 'big_orange_cone' in name:
                label = Label.BIG_ORANGE_CONE
            elif 'orange_cone' in name:
                label = Label.ORANGE_CONE
            else:
                # if not a cone
                continue
            cones.append(Cone(position=pose.position, label=Label(label=label)))



    # def wss_callback(self):
        # Get WSS data
        # msg = WheelSpeeds()

        # Log data retrieval
        # self.get_logger().info('From WSS: %s' % msg)

    # def compute_pose(self):
        
        # Use GNSS, IMU, WSS to process location

        # Publish pose to position topic
        # self.pose_pub.publish(CarPos(position=Point(x=float(self.pos[0, 0]), y=float(self.pos[0, 1])), angle=float(self.pos[0, 2])))

    # Publish to cmd: speed, steering
    # self.cmd_pub.publish(Twist(linear=Vector3(x=float(self.speed)), angular=Vector3(z=-float(steer)))

def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)