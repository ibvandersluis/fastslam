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
import time
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
Q = np.diag([9.0, np.deg2rad(9.0)]) ** 2 # Covariance matrix of measurement noise
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2 # Covariance matrix of observation noise at time t

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.0  # time tick [s]
DThist = [] # List of DTs
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # LM state size [x, y]
N_PARTICLE = 10  # number of particle
THRESHOLD = 0.05 # Likelihood threshold for data association
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

# Definition of variables

    # x: by itself, refers to the state vector [x, y, theta]
    # xd: state with noise

    # u: the control vector [linear velocity, angular velocity]
    # ud: controls with noise

    # z: a set of observations, each containing a vector [distance, angle]
    # z_hat: the predicted observation for a landmark
    # dz: the difference between expected observation and actual observation (z - z_hat)

    # xEst: the estimated true state vector of the vehicle
    # H: Jacobian matrix

    # d: distance (metres)
    # d_sq: distance squared

    # dx: delta x (change in x pos)
    # dy: delta y (change in y pos)
    # dxy: a vector [dx, dy]

    # Particles: a single hypothesis regarding the pose and landmark locations
        # Particle.w: the weight of the particle
        # Particle.x: the robot's pose [x, y, theta]
            # Particle.x[0, 0]: x value of pose
            # Particle.x[1, 0]: y value of pose
            # Particle.x[2, 0]: theta of pose
        # Particle.mu: an array of EKF mean values as x-y coordinates, one for each landmark
        # Particle.sigma: an array of 2x2 covariance matrices for landmark EKFs


# Equations

    # Calculate Qj (measurement covariance)
        # Qj = H @ sigma @ H.T + Q

    # Calculate weight
        # num = np.exp(-0.5 * dz.T @ invQ @ dz)
        # den = 2.0 * np.pi * np.sqrt(np.linalg.det(Qj))

    # w = num / den

# --- CODE ADAPTED FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

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

class Particle:

    def __init__(self):
        """
        Construct a new particle

        :return: Returns nothing
        """
        
        self.w = 1.0 / N_PARTICLE # Particle weight, initialised evenly across particles
        self.x = np.zeros((3, 1)) # State vector [x, y, theta]
        self.mu = np.zeros((0, LM_SIZE)) # Landmark position array (mean of the EKF as x-y coords)
        self.sigma = np.zeros((0, LM_SIZE, LM_SIZE)) # Landmark position covariance array
        self.i = np.zeros((0, 1)) # Counter to evaluate and remove false observations

def fast_slam1(particles, u, z):
    """
    Updates beliefs about position and landmarks using FastSLAM 1.0

    :param particles: An array of particles
    :param u: The controls (velocity and orientation)
    :param z: The observation
    :return: Returns new particles sampled from updated particles according to weight
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
        xEst += particles[i].w * particles[i].x

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst

# STEP 1: PREDICT

def motion_model(x, u):
    """
    Compute predictions for a particle

    :param x: The state vector [x, y, yaw]
    :param u: The input vector [Vt, Wt]
    :return: Returns predicted state vector x
    """
    
    # A 3x3 identity matrix
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # A 3x2 matrix to calculate new x, y, yaw given controls
    B = np.array([[DT * np.cos(x[2, 0]), 0],
                  [DT * np.sin(x[2, 0]), 0],
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
        ud = u + (np.random.randn(1, 2) @ R).T  # Add noise
        particles[i].x = motion_model(particles[i].x, ud) # Run motion model

    return particles

def pi_2_pi(angle):
    """
    Ensure the angle is under +/- PI radians

    :param angle: Angle in radians
    :return: Returns the angle after ensuring it is under +/- PI radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# STEP 2: UPDATE

def observation(xTrue, xd, u, data):
    """
    Record an observation

    :param xTrue: The true state
    :param xd: The dead reckoning state
    :param u: Control vector: linear and angular velocity
    :param data: The landmarks seen by the camera
    :return:
        xTrue - The 'true' state
        z - The observation
        xd - State with noise
        ud - Input with noise
    """
    print('MAKING OBSERVATION')

    # Calc true state
    xTrue = motion_model(xTrue, u)

    # Initialize np array for observed cones
    z = np.zeros_like(data)
    # For each landmark compute distance and angle
    z[:, 0] = np.hypot(data[:, 0], data[:, 1])
    z[:, 1] = pi_2_pi(np.arctan2(data[:, 1], data[:, 0]) - np.pi/2)

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
    for particle in particles:
        # If no landmarks exist yet, add all currently observed landmarks
        if (particle.mu.size == 0):
            # Evaluate sine and cosine values for each observation in z
            s = np.sin(pi_2_pi(particle.x[2, 0] + z[:, 1]))
            c = np.cos(pi_2_pi(particle.x[2, 0] + z[:, 1]))

            # Add new landmark locations to mu
            particle.mu = np.array([particle.x[0, 0] + z[:, 0] * c, particle.x[1, 0] + z[:, 0] * s]).T

            # Distance values
            dpos = np.zeros_like(z)
            dpos[:, 0] = z[:, 0] * c # dx
            dpos[:, 1] = z[:, 0] * s # dy
            d_sq = dpos[:, 0]**2 + dpos[:, 1]**2
            d = np.sqrt(d_sq)

            # Calculate series of H 2x2 Jacobian matrices after the formula
            # H = np.array([[dx / d, dy / d],
            #               [-dy / d_sq, dx / d_sq]])
            dpos_mod = np.flip(dpos, axis=1) # Reverse dpos column order
            dpos_mod[:, 0] = -dpos_mod[:, 0] # Negate dy column
            Ha = dpos/np.vstack(d) # Calculate [dx / d, dy / d]
            Hb = dpos_mod/np.vstack(d_sq) # Calculate [-dy / d_sq, dx / d_sq]
            H = np.vstack((zip(Ha, Hb))).reshape((d.size, 2, 2)) # Weave together

            particle.sigma = np.vstack((particle.sigma, np.linalg.inv(H) @ Q @ np.linalg.inv(H.transpose((0, 2, 1)))))
        else:
            z_hat = np.zeros_like(particle.mu) # Initialise matrix for expected observations
            dpos = particle.mu - particle.x[0:2, 0] # Calculate dx and dy for each landmark
            d_sq = dpos[:, 0]**2 + dpos[:, 1]**2
            z_hat[:, 0] = np.sqrt(d_sq)
            z_hat[:, 1] = pi_2_pi(np.arctan2(dpos[:, 1], dpos[:, 0]) - particle.x[2, 0])

            # Calculate series of H 2x2 Jacobian matrices after the formula
            # H = np.array([[dx / d, dy / d],
            #               [-dy / d_sq, dx / d_sq]])
            dpos_mod = np.flip(dpos, axis=1) # Reverse dpos column order
            dpos_mod[:, 0] = -dpos_mod[:, 0] # Negate dy column
            Ha = dpos/np.vstack((z_hat[:, 0])) # Calculate [dx / d, dy / d]
            Hb = dpos_mod/np.vstack(d_sq) # Calculate [-dy / d_sq, dx / d_sq]
            H = np.vstack((zip(Ha, Hb))) # Weave together
            H = H.reshape(d_sq.size, 2, 2) # Make 3D

            Qj = H @ particle.sigma @ H.transpose((0, 2, 1)) + Q # Calculate covariances

            try:
                invQ = np.linalg.inv(Qj)
            except np.linalg.linalg.LinAlgError:
                print("singular")
                return 1.0

            # For each cone observed, determine data association and add/update
            for iz in range(len(z[:, 0])):
                dz = z_hat - z[iz] # Calculate difference between expectation and observation
                dz[:, 1] = pi_2_pi(dz[:, 1])
                dz = dz.reshape((len(dz[:, 0]), 2, 1)) # reshape as 3D array of 2x1 vectors

                num = np.exp(-0.5 * dz.transpose((0, 2, 1)) @ invQ @ dz)
                den = 2.0 * np.pi * np.sqrt(np.linalg.det(Qj)).reshape((num.size, 1, 1))

                wj = num / den # Calculate likelihoods

                c_max = np.max(wj) # Get max likelihood

                # If the cone probably hasn't been seen before, add the landmark
                if (c_max < THRESHOLD):
                    # Calculate sine and cosine for the landmark
                    s = np.sin(pi_2_pi(particle.x[2, 0] + z[iz, 1]))
                    c = np.cos(pi_2_pi(particle.x[2, 0] + z[iz, 1]))

                    # Add landmark location to mu
                    particle.mu = np.vstack((particle.mu, [particle.x[0, 0] + z[iz, 0] * c, particle.x[1, 0] + z[iz, 0] * s]))

                    dx = z[iz, 0] * c
                    dy = z[iz, 0] * s
                    d_sq = dx**2 + dy**2
                    d = np.sqrt(d_sq) # Get distance
                    Hj = np.array([[dx / d, dy / d],
                                   [-dy / d_sq, dx / d_sq]])
                    Hj = np.linalg.inv(Hj) @ Q @ np.linalg.inv(Hj.T)
                    particle.sigma = np.vstack((particle.sigma, Hj.reshape((1, 2, 2))))

                # If the cone matches a previously seen landmark, update the EKF for that landmark
                else:
                    cj = np.argmax(wj) # Get landmark ID for highest likelihood
                    particle.w *= c_max # Adjust particle weight
                    mu_temp, sigma_temp = update_kf_with_cholesky(particle.mu[cj].reshape((2, 1)),
                                                                  particle.sigma[cj], dz[cj], Q, H[cj])
                    particle.mu[cj] = mu_temp.T # Update landmark EKF mean
                    particle.sigma[cj] = sigma_temp # Replace covariance matrix

    return particles

def update_kf_with_cholesky(mu, sigma, dz, Q_cov, H):
    """
    Update Kalman filter

    :param mu: The mean of a landmark EKF
    :param sigma: The 2x2 covariance of a landmark EKF
    :param dz: The difference between the actual and expected observation
    :param Q_cov: A covariance matrix of process noise
    :param H: Jacobian matrix
    :return:
        mu - New EKF mean as x-y coordinates
        sigma - New EKF covariance matrix
    """
    PHt = sigma @ H.T
    S = H @ PHt + Q_cov

    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T

    mu += W @ dz
    sigma -= W1 @ W1.T

    return mu, sigma

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

    # Create a 1D array of the current particle weights
    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number

    if n_eff.all() < NTH:  # Resampling
        w_cum = np.cumsum(pw) # Cumulative weight is the sum of all particle weights

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
            particles[i].mu = tmp_particles[inds[i]].mu[:, :]
            particles[i].sigma = tmp_particles[inds[i]].sigma[:, :]
            particles[i].w = 1.0 / N_PARTICLE

    return particles

# --- END CODE FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

# ROS 2 Code
class Listener(BaseListener):

    def __init__(self):
        super().__init__('fastslam')

        # Control variables
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

        # Set subscribers
        self.cones_sub = self.create_subscription(ConeArray, '/cones/positions', self.cones_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(IMU, '/peak_gps/imu', self.imu_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/gazebo/cmd_vel', self.control_callback, 10)

        # gets links (all objects) from gazebo
        self.link_sub = self.create_subscription(LinkStates, "/gazebo/link_states", self.link_states_callback, 10)

        self.create_timer(1.0, self.timer_callback)

    def cones_callback(self, msg: ConeArray):
        # Place x y positions of cones into self.capture
        self.capture = np.array([[cone.x, cone.y] for cone in msg.cones])
        # self.capture = self.capture[self.capture[:, 1]>3] # Ignore inputs further than n metres
        print(self.capture)
        # Set time
        global DT, DThist
        cur_time = self.get_clock().now().nanoseconds
        DT = (cur_time - self.timer_last)
        DT /= 1000000000 # Nanoseconds to seconds
        self.timer_last = cur_time # Set timer_last as current nanoseconds
        DThist.append(DT)
        print('DT -- ' + str(DT) + 's')

        # Get observation
        self.xTrue, self.z, self.xDR, self.ud = observation(self.xTrue, self.xDR, self.u, self.capture)

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
                            + ', y: ' + str(particle.mu[i, 1]) + '\n')
            f.write('DThist:\n' + str(DThist))
            f.close()
            self.count = 0

        # Get state estimation
        self.xEst = calc_final_state(self.particles)

        # Boundary check
        self.x_state = self.xEst[0: STATE_SIZE]

        # Store data history
        self.hxEst = np.hstack((self.hxEst, self.x_state))
        self.hxDR = np.hstack((self.hxDR, self.xDR))
        self.hxTrue = np.hstack((self.hxTrue, self.xTrue))

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
        for i in range(len(self.z[:, 0])):
            x = self.xEst[0, 0] # X pos
            y = self.xEst[1, 0] # Y pos
            yaw = self.xEst[2, 0] # Orientation
            d = self.z[i, 0] # Distance from vehicle
            theta = self.z[i, 1] # Angle of observation

            angle = (theta + yaw)

            tx = d * np.cos(angle)
            ty = d * np.sin(angle)

            plt.plot(x + tx, y + ty, "*k", label='Visible Landmarks')

        # point_angle_line(self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0])

        for i in range(N_PARTICLE):
            # Plot landmark estimates as blue X's
            plt.plot(self.particles[i].mu[:, 0], self.particles[i].mu[:, 1], "xb", label='Landmarks')
            # Plot location estimates as red dots
            plt.plot(self.particles[i].x[0, 0], self.particles[i].x[1, 0], ".r", label='Particle Poses')

        # plt.plot(self.hxTrue[0, :], self.hxTrue[1, :], "-b") # Plot xTrue with solid blue line
        # plt.plot(self.hxDR[0, :], self.hxDR[1, :], "-k") # Plot dead reckoning with solid black line
        plt.plot(self.xEst[0], self.xEst[1], "xk", label='Est. Pose') # Plot current xEst as black x
        plt.plot(self.hxEst[0, :], self.hxEst[1, :], "-r", label='Est. Path') # Plot xEst with solid red line
        plt.legend()
        plt.title('FastSLAM 1.0')
        plt.xlabel('X distance (m)')
        plt.ylabel('Y distance (m)')
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