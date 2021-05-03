import os
from itertools import dropwhile
import numpy as np
import math


def is_comment(s):
    """ function to check if a line
         starts with some character.
         Here # for comment
    """
    # return true if a line starts with #
    return s.startswith('#')


def load_mrclam_dataset():
    files_directory=str(os.getcwd())+ '/Data/'
    filename_barcodes = files_directory + "Barcodes.dat"
    barcodes = []
    with open(filename_barcodes, 'r') as f:
        for curline in dropwhile(is_comment, f):
            barcodes_ = curline.split()
            barcodes.append([int(item) for item in barcodes_])
    barcodes = np.array(barcodes)

    filename_landmark_groundtruth = files_directory + "Landmark_Groundtruth.dat"
    landmark_groundtruth = []
    with open(filename_landmark_groundtruth, 'r') as f:
        for curline in dropwhile(is_comment, f):
            landmark_groundtruth_ = curline.split()
            landmark_groundtruth.append([float(item) for item in landmark_groundtruth_])
    n_landmarks = len(landmark_groundtruth)
    landmark_groundtruth = np.array(landmark_groundtruth)

    filename_robot_groundtruth = files_directory + "Robot1_Groundtruth.dat"
    robot_groundtruth = []
    with open(filename_robot_groundtruth, 'r') as f:
        for curline in dropwhile(is_comment, f):
            robot_groundtruth_ = curline.split()
            robot_groundtruth.append([float(item) for item in robot_groundtruth_])
    robot_groundtruth = np.array(robot_groundtruth)

    filename_robot_odometry = files_directory + "Robot1_Odometry.dat"
    robot_odometry = []
    with open(filename_robot_odometry, 'r') as f:
        for curline in dropwhile(is_comment, f):
            robot_odometry_ = curline.split()
            robot_odometry.append([float(item) for item in robot_odometry_])
    robot_odometry = np.array(robot_odometry)

    filename_robot_measurement = files_directory + "Robot1_Measurement.dat"
    robot_measurement = []
    with open(filename_robot_measurement, 'r') as f:
        for curline in dropwhile(is_comment, f):
            robot_measurement_ = curline.split()
            robot_measurement.append([float(item) for item in robot_measurement_])
    robot_measurement = np.array(robot_measurement)

    return barcodes, landmark_groundtruth, n_landmarks, robot_groundtruth, robot_odometry, robot_measurement


def sample_mrclam_dataset(
        robot_groundtruth, robot_odometry, robot_measurement, sample_time=0.02):

    min_time = robot_groundtruth[0][0]
    max_time = robot_groundtruth[-1][0]

    robot_groundtruth[:, 0] -= min_time
    robot_odometry[:, 0] -= min_time
    robot_measurement[:, 0] -= min_time
    max_time -= min_time

    timesteps = math.floor(max_time / sample_time) + 1

    # Robot Groundtruth

    old_data = robot_groundtruth

    k = 0
    t = 0
    i = 0
    p = 0

    nr, nc = old_data.shape
    new_data = np.zeros([timesteps, nc])

    while t <= max_time:
        new_data[k, 0] = t
        while old_data[i, 0] <= t:
            if i == nr - 1:
                break
            i += 1
        if i == 0 or i == nr - 1:
            new_data[k, 1:] = 0
        else:
            p = (t - old_data[i - 1, 0]) / (old_data[i, 0] - old_data[i - 1, 0])
            if nc == 8:
                sc = 3
                new_data[k, 1] = old_data[i, 1]
            else:
                sc = 2

            for c in range(sc, nc + 1):
                if nc == 8 and c >= 6:
                    d = old_data[i, c - 1] - old_data[i - 1, c - 1]
                    if d > math.pi:
                        d -= 2 * math.pi
                    elif d < -math.pi:
                        d = d + 2 * math.pi
                    new_data[k, c - 1] = p * d + old_data[i - 1, c - 1]
                else:
                    new_data[k, c - 1] = p * (old_data[i, c - 1] - old_data[i - 1, c - 1]) + old_data[i - 1, c - 1]

        k = k + 1
        t = t + sample_time

    robot_groundtruth = new_data

    # Robot Odometry

    old_data = robot_odometry

    k = 0
    t = 0
    i = 0
    p = 0

    nr, nc = old_data.shape
    new_data = np.zeros([timesteps, nc])

    while t <= max_time:
        new_data[k, 0] = t
        while old_data[i, 0] <= t:
            if i == nr - 1:
                break
            i += 1
        if i == 0 or i == nr - 1:
            new_data[k, 1:] = old_data[i, 1:]
        else:
            p = (t - old_data[i - 1, 0]) / (old_data[i, 0] - old_data[i - 1, 0])
            if nc == 8:
                sc = 3
                new_data[k, 1] = old_data[i, 1]  # Keep id number
            else:
                sc = 2

            for c in range(sc, nc + 1):
                if nc == 8 and c >= 6:
                    d = old_data[i, c - 1] - old_data[i - 1, c - 1]
                    if d > math.pi:
                        d -= 2 * math.pi
                    elif d < -math.pi:
                        d = d + 2 * math.pi
                    new_data[k, c - 1] = p * d + old_data[i - 1, c - 1]
                else:
                    new_data[k, c - 1] = p * (old_data[i, c - 1] - old_data[i - 1, c - 1]) + old_data[i - 1, c - 1]

        k = k + 1
        t = t + sample_time

    robot_odometry = new_data

    # Robot Measurment

    old_data = robot_measurement
    new_data = old_data
    for j in range(len(old_data)):
        new_data[j, 0] = math.floor(old_data[j, 0] / sample_time + 0.5) * sample_time
    robot_measurement = new_data

    return robot_groundtruth, robot_odometry, robot_measurement, timesteps
