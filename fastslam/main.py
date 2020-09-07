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
import scipy.stats
from copy import deepcopy
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from obr_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates
# from ads_dv_msgs.msg import VCU2AIWheelspeeds

shortcuts.hint()

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2 # Covariance matrix of measurement noise
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2 # Covariance matrix of observation noise at time t

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.0  # time tick [s]
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # LM state size [x, y]
N_PARTICLE = 10  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling
PARTICLE_ITERATION = 0 # n for the nth particle production
UPDATES = 0
ADDS = 0

# Definition of variables

    # x: by itself, refers to the state vector [x, y, theta]
    # u: the control vector [linear velocity, angular velocity]
    # z: a set of observations zi, each containing a vector [distance, angle]
    # z_hat: the predicted observation for a landmark
    # dz: the difference between expected observation and actual observation (z - z_hat)
    # xEst: the estimated true state vector of the vehicle
    # Hv:
    # Hf: 

    # Particles
        # Particle.w: the weight of the particle
        # Particle.x: the x position
        # Particle.y: the y position
        # Particle.theta: the orientation in radians
        # Particle.mu: an array of EKF mean values as x-y coordinates, one for each landmark
        # Particle.sigma: an array of 2x2 covariance matrices for landmark EKFs

# Equations

# Calculate Qj (measurement covariance)
# Qj = H @ sigma @ H.T + Q

def point_angle_line(x, y, theta):
    """
    Plot a line from slope and intercept

    :param rads: The angle of the line in radians
    :param x: The x value of the point
    :point y: The y value of the point
    :return: Nothing
    """
    slope = math.tan(theta)
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
        dx = mu[0, 0] - particle.x # Get relative displacement between landmark and pose
        dy = mu[1, 0] - particle.y
        d_sq = dx**2 + dy**2 # Pythagorean theorem
        d = math.sqrt(d_sq) # Get relative distance from vehicle
        theta = pi_2_pi(math.atan2(dy, dx) - particle.theta) # Get relative angle of observation

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
    c_sq = a**2 + b**2 - 2*a*b * math.cos(theta)
    c = math.sqrt(c_sq)

    return c

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


class Particle:

    def __init__(self):
        """
        Construct a new particle

        :return: Returns nothing
        """
        global PARTICLE_ITERATION
        PARTICLE_ITERATION += 1
        print('Creating particle #' + str(PARTICLE_ITERATION))
        
        self.w = 1.0 / N_PARTICLE # Particle weight
        self.x = 0.0 # X pos
        self.y = 0.0 # Y pos
        self.theta = 0.0 # Orientation
        # Landmark array
        self.mu = np.zeros((0, LM_SIZE + 2)) # Add space for expected distance and angle
        # Landmark position covariance array
        self.sigma = np.zeros((0, LM_SIZE))

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
    weight = 0.0

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].theta
        weight += particles[i].w
    
    print('Weight sum: ' + str(weight))

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst

def motion_model(x, u):
    """
    Compute predictions for a particle

    :param x: The state vector [x, y, yaw]
    :param u: The input vector [Vt, Wt]
    :return: Returns new state vector x
    """
    
    # A 3x3 identity matrix
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # A 3x2 matrix to calculate new x, y, yaw given controls
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
    :param u: An input vector [linear vel, angular vel]
    :return: Returns predictions as particles
    """
    print('PREDICTING PARTICLES')

    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1)) # Creates 3x1 matrix of zeros for x, y, yaw
        px[0, 0] = particles[i].x # Populates top place in matrix with current particle x position
        px[1, 0] = particles[i].y # Populates mid place in matrix with current particle y position
        px[2, 0] = particles[i].theta # Populates bot place in matrix with current particle yaw value
        ud = u + (np.random.randn(1, 2) @ R).T  # add noise
        px = motion_model(px, ud) # Compute predictions using motion model
        particles[i].x = px[0, 0] # Replace particle x pos with predicted value
        particles[i].y = px[1, 0] # Replace particle y pos with predicted value
        particles[i].theta = px[2, 0] # Replace particle yaw with predicted value

        particles[i] = observation_model(particles[i]) # Calculate expected landmark observations

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
    :param xd: The dead reckoning state
    :param u: Control vector: linear and angular velocity
    :param data: The landmarks seen by the camera
    :return:
        xTrue - the true state
        z - the observation
        xd - sate expectation
        ud - Input with noise
    """
    print('MAKING OBSERVATION')

    # Calc true state
    xTrue = motion_model(xTrue, u)

    # Initialize np array for observed cones
    z = np.zeros((2, 0))
    # For each landmark
    for i in range(len(data[:, 0])):
        # Calculate distance d between camera and landmark
        dx = data[i, 0] # X
        dy = data[i, 1] # Y
        d = math.hypot(dx, dy) # Distance
        angle = pi_2_pi(math.atan2(dy, dx)) # Angle
        print('Observation angle: ' + str(angle))
        zi = np.array([d, angle]).reshape(2, 1) # The predicted measurement
        z = np.hstack((z, zi)) # Add prediction to stack of observations

    # Add noise to input
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

    norm = scipy.stats.norm(loc=0.0, scale=1.5) # Generate normal distribution
    threshold = norm.ppf(0.99) # Past this distance gives under 1% probability

    # Get standard deviation for each landmark

    # For each landmark in the observation
    for iz in range(len(z[0, :])):
        # For each particle
        a = time.time()
        for ip in range(N_PARTICLE):
            m = time.time()
            match = False
            # For each landmark
            for lm in range(len(particles[ip].mu[: , 0])):
                j = time.time()
                d = law_of_cos(z[0, iz], particles[ip].mu[lm, 2], z[1, iz] - particles[ip].mu[lm, 3])
                k = time.time()
                print('Calculated d in ' + str(k-j) + 's')
                if (d <= threshold):
                    w = compute_weight(particles[ip], z[:, iz], Q, lm)
                    particles[ip].w *= w
                    particles[ip] = update_landmark(particles[ip], z[:, iz], Q, lm)
                    match = True
                    break
            if (match == False):
                particles[ip] = add_new_landmark(particles[ip], z[:, iz], Q)
            n = time.time()
            print('PARTICLE#' + str(ip) + ' took ' + str(n-m) + 's')
        b = time.time()
        print('OBS#' + str(iz) + ' took ' + str(b-a) + 's')

    return particles

def compute_weight(particle, z, Q_cov, lm_id):
    """
    Compute weight of particles

    :param particle: A particle
    :param z: An observation
    :param Q_cov: The measurement covariance
    :param lm_id: The id of the landmark
    :return: Returns particle weight
    """
    # lm_id = int(z[2]) # Get landmark id from z
    mu = np.array(particle.mu[lm_id, 0:2]).reshape(2, 1) # The pose of a landmark from a particle
    sigma = np.array(particle.sigma[2 * lm_id:2 * lm_id + 2]) # Landmark covariance matrix
    z_hat, Hv, Hf, Qj = compute_jacobians(particle, mu, sigma, Q_cov)

    dz = z[0:2].reshape(2, 1) - z_hat
    dz[1, 0] = pi_2_pi(dz[1, 0])

    try:
        invS = np.linalg.inv(Qj)
    except np.linalg.linalg.LinAlgError:
        print("singular")
        return 1.0

    num = math.exp(-0.5 * dz.T @ invS @ dz)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Qj))

    w = num / den

    return w

def compute_jacobians(particle, mu, sigma, Q_cov):
    """
    Computes Jacobian matrices

    :param particle: A particle
    :param mu: The landmark location
    :param sigma: The covariance matrix for the landmark
    :param Q_cov: A covariance matrix of process noise
    :return:
        z_hat - The relative distance and angle to the landmark
        Hv -
        Hf - 
        Qj - The covariance of measurement noise at time t
    """

    # Compute distance
    dx = mu[0, 0] - particle.x
    dy = mu[1, 0] - particle.y
    d_sq = dx ** 2 + dy ** 2
    d = math.sqrt(d_sq)

    z_hat = np.array([d, pi_2_pi(math.atan2(dy, dx) - particle.theta)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d_sq, -dx / d_sq, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d_sq, dx / d_sq]])

    Qj = Hf @ sigma @ Hf.T + Q_cov

    return z_hat, Hv, Hf, Qj


def add_new_landmark(particle, z, Q_cov):
    """
    Adds a new landmark to a particle

    :param particle: A particle
    :param z: An observation
    :param Q_cov: A covariance matrix of process noise
    :return: A particle
    """
    global ADDS
    ADDS += 1

    r = z[0]
    b = z[1]

    s = math.sin(pi_2_pi(particle.theta + b - math.pi/2))
    c = math.cos(pi_2_pi(particle.theta + b - math.pi/2))

    # Add new lm to array
    particle.mu = np.vstack((particle.mu, [particle.x + r * c, particle.y + r * s, 0.0, 0.0]))

    # covariance
    dx = r * c
    dy = r * s
    d2 = dx**2 + dy**2
    d = math.sqrt(d2) # Get distance
    Gz = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])
    particle.sigma = np.vstack((particle.sigma, np.linalg.inv(Gz) @ Q_cov @ np.linalg.inv(Gz.T)))

    return particle

def update_kf_with_cholesky(mu, sigma, dz, Q_cov, Hf):
    """
    Update Kalman filter

    :param mu:
    :param sigma:
    :param dz: The difference between the actual and expected observation
    :param Q_cov: A covariance matrix of process noise
    :param Hf:
    :return:
        x - 
        P - 
    """
    PHt = sigma @ Hf.T
    S = Hf @ PHt + Q_cov

    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T

    mu = mu + W @ dz
    sigma = sigma - W1 @ W1.T

    return mu, sigma

def update_landmark(particle, z, Q_cov, lm_id):
    """
    Update a landmark

    :param particle: A particle
    :param z: An observation
    :param Q_cov: A covariance matrix of process noise
    :return: A particle
    """
    global UPDATES
    UPDATES += 1

    # lm_id = int(z[2])
    mu = np.array(particle.mu[lm_id, 0:2]).reshape(2, 1)
    sigma = np.array(particle.sigma[2 * lm_id:2 * lm_id + 2, :]) # All columns from this set of 2 rows

    z_hat, Hv, Hf, Qj = compute_jacobians(particle, mu, sigma, Q)

    dz = z[0:2].reshape(2, 1) - z_hat
    dz[1, 0] = pi_2_pi(dz[1, 0])

    mu, sigma = update_kf_with_cholesky(mu, sigma, dz, Q_cov, Hf)

    particle.mu[lm_id, 0:2] = mu.T
    particle.sigma[2 * lm_id:2 * lm_id + 2, :] = sigma # Reassign new covariance matrix

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

    # this creates a 1-d array of the current particle weights
    pw = np.array(pw)

    # each particle is given a resampling significance here of 1/weight^2
    # (the 'effective sampling size') and the sum of these will be a measure
    # of how many current particles are 'doing something useful'
    # The matrix multiply below results in a single number out.
    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(n_eff)

    # have the number of useful particles become too small (i.e., do too
    # many particles have negligible weight)? NTH here is 2/3 the original
    # number of particles.
    if n_eff < NTH:  # resampling
        # generate a vector with the sum of all weights up to the ith particle
        # at index i in the vector
        w_cum = np.cumsum(pw) # Cumulative weight is the sum of all particle weights
        # base will be a vector of length N_PARTICLE with a value for the
        # ith element of i/N_PARTICLE if i is zero-indexed (i.e. you have
        # i = 0, 1, 2, ... N_PARTICLE-1).
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        # resample ID adds some random offset to each element of the base array
        # above. in practice, this means we get an array in units of 1/N_PARTICLE
        # with each element offset by some random fraction of the distance
        # between it and the next multiple of 1/N_PARTICLE. Note that
        # the argument to the rand() function only indicates the shape
        # of an array of random numbers (between 0 and 1) that it will
        # create.
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0
        # go through each particle
        for ip in range(N_PARTICLE):
            # only generate a new particle for indices less than the number of
            # particles already present. resample_id is intended to be a
            # series of n-ile (as in quartile, octile, etc) bins with some
            # random factor, indicating a total hypothetical number of
            # particles thus far generated. w_cum, meanwhile, is supposed
            # to represent the relative probability thus far of particles
            # with indices less than ind having been regenerated. The machinery
            # is starting with the lowest index, generating some number of
            # particles approximately equal to the expected number that it
            # would have had according to the cumulative statistics, then
            # moving on to the next index. This is what the comparison of
            # resample id[ip] with w_cum[ind] is achieving - setting the
            # threshold at which the stepping machinery moves onto the next
            # index to be generated, i.e. the next particle to be replicated.
            # Note that the index ip here is almost irrelevant except as a
            # way of keeping track of how many particles total have been
            # (re)generated. 
            # Some VERY deep and obscure algorithmic trickery is being
            # employed here, this whole section will be VERY non-obvious to
            # people who have not been exposed to a specific description (and
            # probably proof) of the algorithm. Looks like something probably
            # copied out of a paper.
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            # only one index will be generated for each original particle,
            # its value will be some index of one of the particles, but there
            # may (probably will) be duplicates of some indices
            inds.append(ind)

        # each regenerated particle takes its values from the corresponding
        # value of the particle that was indexed by the particular one in inds.
        # for example, the original particle weight array might have looked like:
        # [0.1, 0.3, 0.2, 0.3, 0.1]. Maybe then the generated indices went:
        # [1, 1, 2, 3, 3] (the first and fifth were too low in weight, and
        # the third was lower in weight than the second and fourth). These
        # are the indices that will be selected for the next group of particles.
        # so the system keeps the number of particles the same but redistributes
        # them amongst the available entries by weight.
        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].theta = tmp_particles[inds[i]].theta
            particles[i].mu = tmp_particles[inds[i]].mu[:, :]
            particles[i].sigma = tmp_particles[inds[i]].sigma[:, :]
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
        self.theta = 0.0 # Yaw rate, rad/s

        self.timer_last = self.get_clock().now().nanoseconds
        self.capture = [] # For cone data from snapsot of camera


        # State Vector [x y yaw]
        self.xEst = np.zeros((STATE_SIZE, 1))  # SLAM estimation
        self.xTrue = np.zeros((STATE_SIZE, 1))  # True state
        self.xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

        # History
        self.hxEst = self.xEst
        self.hxTrue = self.xTrue
        self.hxDR = self.xTrue

        # Generate initial particles
        self.particles = [Particle() for _ in range(N_PARTICLE)]

        self.u = np.array([self.v, self.theta]).reshape(2, 1)
        self.ud = None
        self.z = None
        self.count = 0
        self.debug = 0


        # Set publishers
        self.map_pub = self.create_publisher(ConeArray, '/mapping/map', 10)
        self.pose_pub = self.create_publisher(CarPos, '/mapping/position', 10)
        self.cmd_pub = self.create_publisher(Twist, '/gazebo/cmd_vel', 10)

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/gazebo/cmd_vel', self.control_callback, 10)
        # self.wss_sub = self.create_subscription(WheelSpeeds, '/can/ws', self.wss_callback, 10)

        # gets links (all objects) from gazebo
        self.link_sub = self.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_callback, 10)

        self.create_timer(1.0, self.timer_callback)

        # multiple subscribers - not finished
        # cones_sub = message_filters.Subscriber(self, type, '/cones/positions')
        # odom_sub = message_filters.Subscriber(self, type, '/gazebo/odom')
        # gps_sub = message_filters.Subscriber(self, type, '/gps/data')
        # wss_sub = message_filters.Subscriber(self, type, '/wss')

        # ts = message_filters.TimeSynchronizer([points_sub, boxes_sub], 100)
        # ts.registerCallback(self.callback)
        # end multiple subscribers

    def cones_callback(self, msg: ConeArray):
        # Get global variables
        global ADDS
        global UPDATES
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

        print('xEst Angle: ' + str(self.xEst[2, 0]))

        # Run SLAM
        self.particles = fast_slam1(self.particles, self.ud, self.z)

        # Increment counter
        self.count += 1
        # Dump particle lm arrays to text file
        if (self.count >= 1):
            self.debug += 1
            file = 'debug' + str(self.debug) + '.txt'
            f = open(file, 'w')
            pnum = 0
            for particle in self.particles:
                pnum += 1
                f.write('--- Particle #' + str(pnum) + ':\n')
                for i in range(len(particle.mu[:, 0])):
                    f.write('lm #' + str(i + 1) + ' -- x: ' + str(particle.mu[i, 0])
                            + ', y: ' + str(particle.mu[i, 1]) + '\n'
                            + '      -- d: ' + str(particle.mu[i, 2])
                            + ', a: ' + str(particle.mu[i, 3]) + '\n')
            f.write('PARTICLES ADDED: ' + str(ADDS) + '\n')
            f.write('PARTICLES UPDATED: ' + str(UPDATES) + '\n')
            ADDS = 0
            UPDATES = 0
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
        print('Finished cones callback')

    def control_callback(self, msg: Twist):
        str(msg) # For some reason this is needed to access msg.linear.x
        self.v = msg.linear.x
        self.theta = msg.angular.z
        self.u = np.array([self.v, self.theta]).reshape(2, 1)

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

    def timer_callback(self):
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

            angle = (theta + yaw - math.pi/2)

            tx = d * math.cos(angle)
            ty = d * math.sin(angle)

            plt.plot(x + tx, y + ty, "*k")

        # point_angle_line(self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0])

        for i in range(N_PARTICLE):
            # Plot location estimates as red dots
            plt.plot(self.particles[i].x, self.particles[i].y, ".r")
            # Plot landmark estimates as blue X's
            plt.plot(self.particles[i].mu[:, 0], self.particles[i].mu[:, 1], "xb")
            # Plot expected observations of landmarks
            # for particle in self.particles:
            #     x = particle.x
            #     y = particle.y
            #     for lm in particle.lm:
            #         d = lm[2]
            #         theta = lm[3]
            #         angle = theta + particle.theta
            #         tx = d * math.cos(angle)
            #         ty = d * math.sin(angle)

            #         plt.plot(x + tx, y + ty, "xg")                    

        plt.plot(self.hxTrue[0, :], self.hxTrue[1, :], "-b") # Plot xTrue with solid blue line
        plt.plot(self.hxDR[0, :], self.hxDR[1, :], "-k") # Plot dead reckoning with solid black line
        plt.plot(self.hxEst[0, :], self.hxEst[1, :], "-r") # Plot xEst with solid red line
        plt.plot(self.xEst[0], self.xEst[1], "xk") # Plot current xEst as black x
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

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