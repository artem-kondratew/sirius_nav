import numpy as np
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

# plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    # calculate and plot covariance ellipse
    covariance = sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    # get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    # get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigval = eigenvals[min_ind]

    # chi-square value for sigma confidence interval
    chi_square_scale = 2.2789

    # calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chi_square_scale * max_eigval)
    height = 2 * np.sqrt(chi_square_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    # generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle / np.pi * 180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)


def normalize_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution

    x, y, theta = mu.flatten()
    
    ut = odometry['t']
    ur1 = odometry['r1']
    ur2 = odometry['r2']

    f = np.array([
        [x + ut * np.cos(theta + ur1)],
        [y + ut * np.sin(theta + ur1)],
        [normalize_angle(theta + ur1 + ur2)],
    ])

    Jf = np.array([
        [1, 0, -ut*np.sin(theta+ur1)],
        [0, 1, ut*np.cos(theta+ur1)],
        [0, 0, 1],
    ])

    Ju = np.array([
        [np.cos(theta+ur1), -ut*np.sin(theta+ur1), 0],
        [np.sin(theta+ur1), ut*np.cos(theta+ur1), 0],
        [0, 1, 1],
    ])

    mu = f

    Q = np.eye(mu.shape[0]) * 0.2
    
    sigma = Jf @ sigma @ Jf.T + Ju @ Q @ Ju.T

    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x, y = mu[:2].flatten()

    # measured landmark ids and ranges
    ids = sensor_data['id']
    z = np.array(sensor_data['range']).reshape(-1, 1)

    h = np.zeros_like(z)
    Jh = np.zeros((len(ids), len(mu)))

    for i, id in enumerate(ids):
        landmark = np.array(landmarks[id]).reshape(-1, 1)
        dist = np.linalg.norm(mu[:2] - landmark)
        h[i, 0] = dist
        Jh[i] = [(x - landmark[0, 0]) / dist, (y - landmark[1, 0]) / dist, 0]

    R = np.eye(len(ids)) * 0.5

    K = sigma @ Jh.T @ np.linalg.inv(Jh @ sigma @ Jh.T + R)

    mu = mu + K @ (z - h)
    mu[2, 0] = normalize_angle(mu[2, 0])

    I = np.eye(sigma.shape[0])
    sigma = (I - K @ Jh) @ sigma

    return mu, sigma


def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    # initialize belief
    mu = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)

    sigma = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    # run kalman filter
    for timestep in range(len(sensor_readings) // 2):

        # plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        # perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep, 'odometry'], mu, sigma)

        # perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

        # exit(0)

    plt.show(block=True)


if __name__ == "__main__":
    main()
