from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0, 0, 0] with an empty map

    particles = []

    for _ in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0, 0, 0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # "move" each particle according to the odometry measurements plus sampled noise to generate new particle set

    for p in particles:
        x, y, theta = p['x'], p['y'], p['theta']
   
        a1, a2, a3, a4 = noise
        disp_rot_1 = a1 * np.abs(delta_rot1) + a2 * delta_trans
        disp_rot_2 = a1 * np.abs(delta_rot2) + a2 * delta_trans
        disp_tran = a3 * delta_trans + a4 * (np.abs(delta_rot1) + np.abs(delta_rot2))

        dr1_noised = delta_rot1 + np.random.normal(loc=0, scale=disp_rot_1)
        dr2_noised = delta_rot2 + np.random.normal(loc=0, scale=disp_rot_2)
        dt_noised = delta_trans + np.random.normal(loc=0, scale=disp_tran)

        x_n = x + dt_noised * np.cos(theta + dr1_noised)
        y_n = y + dt_noised * np.sin(theta + dr1_noised)
        theta_n = theta + dr1_noised + dr2_noised

        p['x'], p['y'], p['theta'] = x_n, y_n, normalize_angle(theta_n)

        p['history'].append([x_n, y_n])


def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    p_theta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - p_theta

    h = np.array([meas_range_exp, normalize_angle(meas_bearing_exp)])

    # Compute the Jacobian h_j of the measurement function h
    # wrt the landmark location

    h_j = np.zeros((2, 2))
    h_j[0, 0] = (lx - px) / h[0]
    h_j[0, 1] = (ly - py) / h[0]
    h_j[1, 0] = (py - ly) / (h[0] ** 2)
    h_j[1, 1] = (lx - px) / (h[0] ** 2)

    return h, h_j


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    R = np.array([[0.1, 0],
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']
        particle['weight'] = 1.0

        px = particle['x']
        py = particle['y']
        p_theta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            z = np.array([meas_range, meas_bearing])

            if not landmark['observed']:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                
                landmark['mu'][0] = px + meas_range * np.cos(p_theta + meas_bearing)
                landmark['mu'][1] = py + meas_range * np.sin(p_theta + meas_bearing)

                _, H = measurement_model(particle, landmark)
                
                Hinv = np.linalg.inv(H)
                landmark['sigma'] = Hinv @ R @ Hinv.T

                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...

                h, H = measurement_model(particle, landmark)

                sigma = landmark['sigma']

                Q = H @ sigma @ H.T + R
                
                K = sigma @ H.T @ np.linalg.inv(Q)

                d = z - h
                d[1] = normalize_angle(d[1])

                landmark['mu'] += K @ (z - h)

                I = np.eye(sigma.shape[0])
                landmark['sigma'] = (I - K @ H) @ sigma

                w = 1/np.sqrt(2*np.pi*np.linalg.det(Q)) * np.exp(-1/2 * (z - h).T @ np.linalg.inv(Q) @ (z - h))

                particle['weight'] *= w


    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.
    # distance between pointers

    step = 1.0/len(particles)

    u = np.random.uniform(0,step)

    c = particles[0]['weight']
    i = 0

    new_particles = []

    for _ in particles:
        while u > c:
            i += 1
            c += particles[i]['weight']

        new_particle = copy.deepcopy(particles[i])
        new_particle['weight'] = 1.0 / len(particles)
        new_particles.append(new_particle)

        u += step
    return new_particles


def main():

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    # run FastSLAM
    for timestep in range(len(sensor_readings) // 2):

        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        # plot filter state
        plot_state(particles, landmarks)

        # calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show(block=True)


if __name__ == "__main__":
    main()
