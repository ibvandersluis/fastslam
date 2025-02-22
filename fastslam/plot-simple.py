import numpy as np
import matplotlib.pyplot as plt

blue_cones = np.array([(-16.7219, 9.56934),
                       (-12.7553, 12.49),
                       (-6.89062, 13.0757),
                       (9.62956, 15.8379),
                       (12.492, 17.0404),
                       (15.1389, 16.5025),
                       (19.5034, 13.6361),
                       (20.8922, 11.7012),
                       (17.5221, 15.3812),
                       (21.4761, 8.79574),
                       (20.9992, 5.27662),
                       (19.9925, 2.24053),
                       (19.0983, 0.0),
                       (17.1824, -3.23994),
                       (11.114, -6.74408),
                       (-4.06552, 13.3637),
                       (14.2831, -5.26437),
                       (8.25363, -8.53889),
                       (5.06185, -10.1551),
                       (1.42086, -11.9634),
                       (-2.4975, -14.0305),
                       (-5.74864, -16.1217),
                       (-9.34841, -17.1551),
                       (-12.2114, -16.6459),
                       (-14.4625, -14.9249),
                       (-16.2427, -13.276),
                       (-0.131239, 13.3125),
                       (-18.1431, -11.086),
                       (-18.6174, -7.56085),
                       (-18.9382, -5.15509),
                       (-18.5558, -2.56017),
                       (-18.1206, 0.0),
                       (-17.7841, 3.04246),
                       (-17.8432, 6.27091),
                       (-15.1864, 11.7137),
                       (3.50416, 13.7245),
                       (7.13676, 14.727)])

yellow_cones = np.array([(-12.2184, 7.60803),
                         (7.14787, 9.92656),
                         (10.4312, 10.799),
                         (12.9655, 11.8014),
                         (14.9652, 11.2833),
                         (16.6054, 9.12035),
                         (-6.90998, 8.44454),
                         (16.7063, 6.10772),
                         (15.876, 3.47906),
                         (15.106, 1.5027),
                         (13.6765, 0.0),
                         (9.41953, -2.84739),
                         (12.0732, -1.43122),
                         (7.26282, -3.9408),
                         (4.65159, -5.15509),
                         (1.78774, -6.66723),
                         (-1.97969, -8.56329),
                         (-4.01695, 8.50322),
                         (-5.18123, -10.5555),
                         (-7.57043, -12.1125),
                         (-9.76388, -12.7081),
                         (-12.0338, -11.6718),
                         (-13.9298, -8.98291),
                         (-14.1575, -5.77329),
                         (-14.0043, -2.62998),
                         (-13.6087, 0.0),
                         (-13.3478, 3.05712),
                         (-0.059759, 8.37591),
                         (-13.3455, 5.78808),
                         (3.5151, 8.6968)])

orange_cones = np.array([(-9.61314, 13.0),
                         (-9.99934, 12.989),
                         (-9.62148, 8.39323),
                         (-9.98667, 8.39348)])

plt.plot(blue_cones[:, 0], blue_cones[:, 1], '^b', label='Blue cones')
plt.plot(yellow_cones[:, 0], yellow_cones[:, 1], '^y', label='Yellow cones')
plt.plot(orange_cones[:, 0], orange_cones[:, 1],
         marker='x', c='orange', linestyle='None', label='Starting line')

plt.legend()
plt.title('Simple Track')
plt.xlabel('X distance (m)')
plt.ylabel('Y distance (m)')

plt.show()