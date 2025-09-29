import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

# add random seed for generating comparable pseudo random numbers
np.random.seed(123)

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)


def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        # make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    # calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]


def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # "move" each particle according to the odometry measurements plus sampled noise to generate new particle set
    new_particles = []

    for p in particles:
        x, y, theta = p['x'], p['y'], p['theta']
   
        a1, a2, a3, a4 = noise
        disp_rot_1 = a1 * np.abs(delta_rot1) + a2 * delta_trans
        disp_rot_2 = a1 * np.abs(delta_rot2) + a2 * delta_trans
        disp_tran = a3 * delta_trans + a4 * (np.abs(delta_rot1) + np.abs(delta_rot2))

        dr1_noised = delta_rot1 + np.random.normal(loc=0, scale=np.sqrt(disp_rot_1))
        dr2_noised = delta_rot2 + np.random.normal(loc=0, scale=np.sqrt(disp_rot_2))
        dt_noised = delta_trans + np.random.normal(loc=0, scale=np.sqrt(disp_tran))

        x_n = x + dt_noised * np.cos(theta + dr1_noised)
        y_n = y + dt_noised * np.sin(theta + dr1_noised)
        theta_n = theta + dr1_noised + dr2_noised

        new_particles.append({'x': x_n, 'y': y_n, 'theta': theta_n})

    return new_particles


def normal_distribution(x, mean, std):
    return 1/std/np.sqrt(2*np.pi) * np.exp(-(x - mean)**2 / 2 / std**2)


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    for p in particles:
        robot_coords = np.array([p['x'], p['y']]).reshape(-1, 1)
        likelihood = 1
        for id, measurement in zip(ids, ranges):
            landmark_coords = np.array(landmarks[id]).reshape(-1, 1)
            dist = np.linalg.norm(robot_coords - landmark_coords)
            likelihood *= normal_distribution(dist, measurement, sigma_r)
        weights.append(likelihood)

    normalizer = sum(weights)
    weights = weights / normalizer

    return weights


def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    step = 1 / len(weights)
    r = np.random.uniform(0, step)

    uarr = [r + i * step for i in range(len(weights))]

    sum = 0
    i = 0

    for u in uarr:
        while u > sum:
           sum += weights[i]
           i += 1
        new_particles.append(particles[i-1])

    return new_particles


def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    # initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    # run particle filter
    for timestep in range(len(sensor_readings) // 2):

        # plot the current state
        plot_state(particles, landmarks, map_limits)

        # predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        # resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show(block=True)


if __name__ == "__main__":
    main()
