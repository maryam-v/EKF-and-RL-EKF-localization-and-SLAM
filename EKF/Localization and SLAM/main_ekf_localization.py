import os
import numpy as np
import math
from prepare_data import load_mrclam_dataset
from prepare_data import sample_mrclam_dataset
from get_observations import get_observations
from utils import con_bear
from animate_mrclam_dataset import animate_mrclam_dataset


def main_ekf_localization(
        barcodes, landmark_groundtruth,
        robot_groundtruth, robot_odometry, robot_measurement,
        deltaT, alphas, Q_t
):
    # add pose estimate matrix to Robots
    # data will be added to this as the program runs

    Robot_Est = np.zeros([robot_groundtruth.shape[0], 4])
    error = np.zeros([robot_groundtruth.shape[0], 3]).reshape(-1, 3, 1)

    # initialize time, and pose estimate
    start = 599  # start index is set to 599 because earlier data was found to cause problems
    t = robot_groundtruth[start, 0]  # set start time

    # set starting pose mean to pose groundtruth at start time
    poseMean = np.array([robot_groundtruth[start, 1],
                         robot_groundtruth[start, 2],
                         robot_groundtruth[start, 3]]).reshape(3, 1)
    poseCov = np.array([[0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01]])

    # tracks which measurement is next received
    measurementIndex = 0

    # set up map between barcodes and landmark IDs
    codeDict = dict()
    j = 0
    for i in barcodes[:, 1]:
        codeDict[i] = barcodes[j, 0]
        j += 1

    # advance measurement index until the next measurement time is greater than the starting time
    while robot_measurement[measurementIndex, 0] < t - 0.05:
        measurementIndex += 1

    # loop through all odometry and measurement samples updating the robot's pose estimate with each step reference table 7.2 in Probabilistic Robotics
    for i in range(start, robot_groundtruth.shape[0]):
        theta = poseMean[2][0]
        # update time
        t = robot_groundtruth[i, 0]

        u_t = np.array([robot_odometry[i, 1], robot_odometry[i, 2]]).reshape(2, 1)
        rot = deltaT * u_t[1][0]
        halfRot = rot / 2;
        trans = u_t[0][0] * deltaT;

        # calculate the movement Jacobian
        G_t = np.array([[1, 0, -trans * math.sin(theta + halfRot)],
                        [0, 1, trans * math.cos(theta + halfRot)],
                        [0, 0, 1]])

        # calculate motion covariance in control space
        M_t = np.array([[(alphas[0] * abs(u_t[0][0]) + alphas[1] * abs(u_t[1][0])) ** 2, 0],
                        [0, (alphas[2] * abs(u_t[0][0]) + alphas[3] * abs(u_t[1][0])) ** 2]])
        # M_t = np.array([[alphas[0] * abs(u_t[0][0])**2 + alphas[1] * abs(u_t[1][0])**2,0],
        #                [0,alphas[2] * abs(u_t[0][0])**2 + alphas[3] * abs(u_t[1][0])**2]])

        # calculate Jacobian to transform motion covariance to state space
        V_t = np.array([[math.cos(theta + halfRot), -0.5 * math.sin(theta + halfRot)],
                        [math.sin(theta + halfRot), 0.5 * math.cos(theta + halfRot)],
                        [0, 1]])

        # calculate pose update
        poseUpdate = np.array([trans * math.cos(theta + halfRot),
                               trans * math.sin(theta + halfRot),
                               rot]).reshape(3, 1)

        # calculate estimated pose mean
        poseMeanBar = poseMean + poseUpdate

        # get measurements for the current timestep
        poseCovBar = np.matmul(np.matmul(G_t, poseCov), G_t.T) + np.matmul(np.matmul(V_t, M_t), V_t.T)

        # get measurements for the current timestep, if any exist
        z, measurementIndex = get_observations(robot_measurement, t, measurementIndex, codeDict)

        # create two matrices for expected measurement and measurement covariance
        S = np.zeros([z.shape[1], 3, 3])
        zHat = np.zeros([3, z.shape[1]])

        # if any measurements are available
        if z[2][0] > 1:
            for k in range(0, z.shape[1]):
                j = z[2][k]

                # get coordinates of the measured landmark
                m = landmark_groundtruth[int(j) - 1, 1:3]

                # compute the expected measurement
                xDist = m[0] - poseMeanBar[0][0]
                yDist = m[1] - poseMeanBar[1][0]
                q = xDist ** 2 + yDist ** 2

                # constrains expected bearing to between 0 and 2*pi
                pred_bear = con_bear(math.atan2(yDist, xDist) - poseMeanBar[2][0])
                zHat[:, k] = [math.sqrt(q), pred_bear, j]

                # calculate Jacobian of the measurement model
                H = np.array([[-1 * (xDist / math.sqrt(q)), -1 * (yDist / math.sqrt(q)), 0],
                              [yDist / q, -xDist / q, -1],
                              [0, 0, 0]])

                # compute S
                S[k, :, :] = np.matmul(np.matmul(H, poseCovBar), H.T) + Q_t

                # compute Kalman gain
                K = np.matmul(np.matmul(poseCov, H.T), np.linalg.inv(S[k, :, :]))

                # update pose mean and covariance estimates
                poseMeanBar = poseMeanBar + np.matmul(K, (z[:, k] - zHat[:, k])).reshape(3, 1)
                poseCovBar = np.matmul(np.identity(3) - np.matmul(K, H), poseCovBar)

        # update pose mean and covariance constrains heading to between 0 and 2*pi
        poseMean = poseMeanBar
        poseMean[2][0] = con_bear(poseMean[2][0])
        poseCov = poseCovBar

        # add pose mean to estimated position vector
        Robot_Est[i, :] = [t, poseMean[0][0], poseMean[1][0], poseMean[2][0]]

        # calculate error between mean pose and groundtruth for testing only
        groundtruth = np.array([robot_groundtruth[i, 1],
                                robot_groundtruth[i, 2],
                                robot_groundtruth[i, 3]]).reshape(3, 1)

        error[i, :, :] = groundtruth - poseMean
        
    # computes euclidean loss between robot's estimated path and ground truth ignores bearing error
    path_loss = 0
    for i in range(start, robot_groundtruth.shape[0]):
        x_diff = robot_groundtruth[i][1] - Robot_Est[i, 1]
        y_diff = robot_groundtruth[i][2] - Robot_Est[i, 2]
        err = math.sqrt(x_diff ** 2 + y_diff ** 2)
        path_loss = path_loss + err
    path_loss = path_loss / len(range(start, robot_groundtruth.shape[0]))
    print('average path_loss: ', path_loss)

    return K, Robot_Est, error, path_loss


if __name__ == '__main__':
    deltaT = .02
    alphas = np.array([.2, .03, .09, .08])  # robot-dependent motion noise parameters

    # robot-dependent sensor noise parameters
    sigma_range = 2
    sigma_bearing = 3
    sigma_id = 1
    Q_t = np.array([[sigma_range ** 2, 0, 0], [0, sigma_bearing ** 2, 0], [0, 0, sigma_id ** 2]])

    barcodes, landmark_groundtruth, n_landmarks, \
    robot_groundtruth, robot_odometry, robot_measurement = \
        load_mrclam_dataset(files_directory=str(os.getcwd()) + '/Data/')

    robot_groundtruth, robot_odometry, robot_measurement, timesteps = \
        sample_mrclam_dataset(
            robot_groundtruth,
            robot_odometry,
            robot_measurement,
            sample_time=deltaT
        )

    K, Robot_Est, error, path_loss = main_ekf_localization(
        barcodes, landmark_groundtruth,
        robot_groundtruth, robot_odometry, robot_measurement,
        deltaT, alphas, Q_t
    )

    animate_mrclam_dataset(
        robot_groundtruth,
        Robot_Est,
        landmark_groundtruth,
        timesteps,
        robot_measurement
    )
