import time
import numpy as np

def pi_2_pi(angle):
    """
    Ensure the angle is under +/- PI radians

    :param angle: Angle in radians
    :return: Returns the angle after ensuring it is under +/- PI radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

data = np.array([[1.0, 1.0],
                 [2.0, 2.0],
                 [3.0, 3.0],
                 [4.0, 4.0]])
a = time.time()
# Initialize np array for observed cones
# For each landmark
# for i in range(len(data[:, 0])):
#     # Calculate distance d between camera and landmark
#     dx = data[i, 0] # X
#     dy = data[i, 1] # Y
#     d = np.hypot(dx, dy) # Distance
#     theta = pi_2_pi(np.arctan2(dy, dx)) # Angle
#     print('Observation angle: ' + str(theta))
#     zi = np.array([d, theta]).reshape(2, 1) # The predicted measurement
#     z = np.hstack((z, zi)) # Add prediction to stack of observations
z = np.zeros_like(data)
# For each landmark compute distance and angle
z[:, 0] = np.hypot(data[:, 0], data[:, 1])
z[:, 1] = pi_2_pi(np.arctan2(data[:, 0], data[:, 1]))
b = time.time()
print('OBSERVATION TIME: ' + str(b-a))
print(z.T)