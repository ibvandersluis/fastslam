"""
Starter code from Atsushi Sakai,
which I have since modified for my use in main.py
"""

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

import sys

import message_filters
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from helpers.listener import BaseListener
from helpers import shortcuts
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from brookes_msgs.msg import Cone, CarPos, ConeArray, IMU, Label
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Twist, Vector3
from gazebo_msgs.msg import LinkStates

# STEP 1: PREDICT

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2 # Covariance matrix of process noise
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2 # Covariance matrix of observation noise at time t

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # LM state size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

show_animation = True

class Particle:

    def __init__(self, n_landmark):
        """
        Construct a new particle

        :param n_landmark: The landmark number
        :return: Returns nothing
        """
        
        self.w = 1.0 / N_PARTICLE # Particle weight?
        self.x = 0.0 # X pos
        self.y = 0.0 # Y pos
        self.yaw = 0.0 # Orientation
        # Landmark x-y positions
        self.lm = np.zeros((n_landmark, LM_SIZE))
        # Landmark position covariance
        self.lmP = np.zeros((n_landmark * LM_SIZE, LM_SIZE))

def fast_slam1(particles, u, z):
    """
    Updates beliefs about position and landmarks using FastSLAM 1.0

    :param particles:
    :param u: Ut = [Vt, Wt], the velocity and orientation at a given time
    :param z: Zt = [Xt, Yt], the X-Y position at a given time
    :return: Returns updated particles (position and landmarks)
    """

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
    xEst = np.zeros((STATE_SIZE, 1)) # Empty state vector for: y, y, yaw

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    #  print(xEst)

    return xEst

def calc_input(time):
    """
    Calculate input vector

    :param time: The elapsed time in seconds
    :return: An input vector u containing the velocity and orientation
    """
    if time <= 3.0:  # wait at first
        v = 0.0
        yaw_rate = 0.0
    else:
        v = 1.0  # [m/s]
        yaw_rate = 0.1  # [rad/s]

    u = np.array([v, yaw_rate]).reshape(2, 1)

    return u

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

# END OF SNIPPET

# N_LM = 0
# particles = [Particle(N_LM) for i in range(N_PARTICLE)] # Generate array of 100 particles
# time= 0.0
# v = 1.0  # [m/s]
# yawrate = 0.1  # [rad/s]
# u = np.array([v, yawrate]).reshape(2, 1)
# history = []
# while SIM_TIME >= time:
#     time += DT
#     particles = predict_particles(particles, u)
#     history.append(deepcopy(particles))

# STEP 2: UPDATE

def observation(xTrue, xd, u, rfid):
    """
    Record an observation

    :param xTrue: The true state
    :param xd: 
    :param u: Velocity and Yaw
    :param rfid: The true map of landmarks
    :return:
        xTrue - the true state
        z - the observation
        xd - sate expectation
        ud - Input with noise
    """
    # calc true state
    xTrue = motion_model(xTrue, u)

    # add noise to range observation
    z = np.zeros((3, 0))
    # For each landmark
    for i in range(len(rfid[:, 0])):
        # Get true distance d between pose and landmark
        dx = rfid[i, 0] - xTrue[0, 0]
        dy = rfid[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        # If the object is close enough to sense:
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_with_noise = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, pi_2_pi(angle_with_noise), i]).reshape(3, 1) # The predicted measurement
            z = np.hstack((z, zi)) # add prediction to stack of observations

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def update_with_observation(particles, z):
    """
    Updates particles using observation

    :param particles: An array of particles
    :param z: An observation [Xt, Yt]
    :return: Returns updated particles
    """
    for iz in range(len(z[0, :])):

        landmark_id = int(z[2, iz])

        for ip in range(N_PARTICLE):
            # new landmark
            if abs(particles[ip].lm[landmark_id, 0]) <= 0.01:
                particles[ip] = add_new_landmark(particles[ip], z[:, iz], Q)
            # known landmark
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
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
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

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf


def add_new_landmark(particle, z, Q_cov):
    """
    Adds a new landmark to [a particle?]

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

    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s

    # covariance
    dx = r * c
    dy = r * s
    d2 = dx**2 + dy**2
    d = math.sqrt(d2) # Get distance
    Gz = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])
    particle.lmP[2 * lm_id:2 * lm_id + 2] = np.linalg.inv(
        Gz) @ Q_cov @ np.linalg.inv(Gz.T)

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
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle

# END OF CODE SNIPPET #



# # Setting up the landmarks
# RFID = np.array([[10.0, -2.0],
#                 [15.0, 10.0]])
# N_LM = RFID.shape[0]

# # Initialize 1 particle
# N_PARTICLE = 1
# particles = [Particle(N_LM) for i in range(N_PARTICLE)]

# xTrue = np.zeros((STATE_SIZE, 1))
# xDR = np.zeros((STATE_SIZE, 1))

# print("initial weight", particles[0].w)

# xTrue, z, _, ud = observation(xTrue, xDR, u, RFID)
# # Initialize landmarks
# particles = update_with_observation(particles, z)
# print("weight after landmark initialization", particles[0].w)
# particles = update_with_observation(particles, z)
# print("weight after update ", particles[0].w)

# particles[0].x = -10
# particles = update_with_observation(particles, z)
# print("weight after wrong prediction", particles[0].w)

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
# END OF SNIPPET #



# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# N_PARTICLE = 100
# particles = [Particle(N_LM) for i in range(N_PARTICLE)]
# x_pos = []
# w = []
# for i in range(N_PARTICLE):
#     particles[i].x = np.linspace(-0.5,0.5,N_PARTICLE)[i]
#     x_pos.append(particles[i].x)
#     particles[i].w = gaussian(i, N_PARTICLE/2, N_PARTICLE/20)
#     w.append(particles[i].w)


# # Normalize weights
# sw = sum(w)
# for i in range(N_PARTICLE):
#     w[i] /= sw

# particles, new_indices = resampling(particles)
# x_pos2 = []
# for i in range(N_PARTICLE):
#     x_pos2.append(particles[i].x)

# # Plot results
# fig, ((ax1,ax2,ax3)) = plt.subplots(nrows=3, ncols=1)
# fig.tight_layout()
# ax1.plot(x_pos,np.ones((N_PARTICLE,1)), '.r', markersize=2)
# ax1.set_title("Particles before resampling")
# ax1.axis((-1, 1, 0, 2))
# ax2.plot(w)
# ax2.set_title("Weights distribution")
# ax3.plot(x_pos2,np.ones((N_PARTICLE,1)), '.r')
# ax3.set_title("Particles after resampling")
# ax3.axis((-1, 1, 0, 2))
# fig.subplots_adjust(hspace=0.8)
# plt.show()

# plt.figure()
# plt.hist(new_indices)
# plt.xlabel("Particles indices to be resampled")
# plt.ylabel("# of time index is used")
# plt.show()

# code from main function
def pr_main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [15.0, 15.0],
                     [10.0, 20.0],
                     [3.0, 15.0],
                     [-5.0, 20.0],
                     [-5.0, 5.0],
                     [-10.0, 15.0]
                     ])
    # numpy shape attribute is the dimensions of a matrix
    n_landmark = RFID.shape[0] # the number of coordinate pairs on RFID

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))  # SLAM estimation
    xTrue = np.zeros((STATE_SIZE, 1))  # True state
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # History
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]

    while SIM_TIME >= time:
        time += DT # Increment time
        u = calc_input(time) # Set input based on time

        # Get observation
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        # Run SLAM
        particles = fast_slam1(particles, ud, z)

        # Get state estimation
        xEst = calc_final_state(particles)

        # What does this do??
        x_state = xEst[0: STATE_SIZE]

        # Store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # Pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event', lambda event:
                [exit(0) if event.key == 'escape' else None])
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")

            for i in range(N_PARTICLE):
                plt.plot(particles[i].x, particles[i].y, ".r")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :], hxDR[1, :], "-k")
            plt.plot(hxEst[0, :], hxEst[1, :], "-r")
            plt.plot(xEst[0], xEst[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

# --- END CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---