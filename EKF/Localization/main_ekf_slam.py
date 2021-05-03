import os
import numpy as np
import math
from prepare_data import load_mrclam_dataset
from prepare_data import sample_mrclam_dataset
from get_observations import get_observations
from utils import con_bear
from animate_mrclam_dataset import animate_mrclam_dataset


def main_ekf_slam(
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

    # initialize state mean with the groundtruth pose at start time
    stateMean = np.array([robot_groundtruth[start, 1],
                          robot_groundtruth[start, 2],
                          robot_groundtruth[start, 3]]).reshape(3, 1)

    # reshapes stateMean to 1x(2*n_landmarks+3) and remaps motion updates to new state vector
    F_x = np.hstack([np.identity(3), np.zeros([3, 2 * n_landmarks])])
    stateMean = np.matmul(F_x.T, stateMean)

    # initialize diagonal of pose covariance with small, nonzero values since we have a good estimate of the starting pose
    stateCov = np.zeros([2 * n_landmarks + 3, 2 * n_landmarks + 3])
    stateCov[:3, :3] = 0.001

    # initialize landmark variances to large values since we have no prior information on the locations of the landmarks
    for i in range(3, n_landmarks * 2 + 3):
        stateCov[i, i] = 10

    measurementIndex = 0

    # set up map between barcodes and landmark IDs barcode values are NOT the same as landmark IDs
    codeDict = dict()
    j = 0
    for i in barcodes[:, 1]:
        codeDict[i] = barcodes[j, 0]
        j += 1

    # loop through all odometry and measurement samples updating the robot's pose estimate with each step reference table 7.2 in Probabilistic Robotics
    for i in range(start, robot_groundtruth.shape[0]):

        # update time
        t = robot_groundtruth[i, 0]

        # get control inputs at current timestep
        u_t = np.array([robot_odometry[i, 1], robot_odometry[i, 2]])

        # update robot bearing to last timestep's estimate
        theta = stateMean[2][0]


        # calculate rotation and translation over last timestep
        rot = deltaT * u_t[1]
        halfRot = rot / 2
        trans = u_t[0] * deltaT


        # calculate pose update in world frame
        poseUpdate = np.array([trans * math.cos(theta + halfRot),
                               trans * math.sin(theta + halfRot),
                               rot]).reshape(3, 1)

        # updated state mean and constrain bearing to +/- pi
        stateMeanBar = stateMean + np.matmul(F_x.T, poseUpdate)
        stateMeanBar[2][0] = con_bear(stateMeanBar[2][0])


        # calculate the movement Jacobian
        g_t = np.array([[0, 0, -trans * math.sin(theta + halfRot)],
                        [0, 0, trans * math.cos(theta + halfRot)],
                        [0, 0, 0]])

        G_t = np.identity(2 * n_landmarks + 3) + np.matmul(np.matmul(F_x.T, g_t), F_x)


        # calculate motion covariance in control space
        M_t = np.array([[(alphas[0] * abs(u_t[0]) + alphas[1] * abs(u_t[1])) ** 2, 0],
                        [0, (alphas[2] * abs(u_t[0]) + alphas[3] * abs(u_t[1])) ** 2]])
        # M_t = np.array([[alphas[0] * abs(u_t[0][0])**2 + alphas[1] * abs(u_t[1][0])**2,0],
        #                [0,alphas[2] * abs(u_t[0][0])**2 + alphas[3] * abs(u_t[1][0])**2]])


        # calculate Jacobian to transform motion covariance to state space
        V_t = np.array([[math.cos(theta + halfRot), -0.5 * math.sin(theta + halfRot)],
                        [math.sin(theta + halfRot), 0.5 * math.cos(theta + halfRot)],
                        [0, 1]])


        # update state covariance
        R_t = np.matmul(np.matmul(V_t, M_t), V_t.T)
        stateCovBar = np.matmul(np.matmul(G_t, stateCov), G_t.T) + np.matmul(np.matmul(F_x.T, R_t), F_x)


        # get measurements for current timestep
        z, measurementIndex = get_observations(robot_measurement, t, measurementIndex, codeDict)

        # if any measurements are available
        if z[2][0] > 1:
            zHat = np.zeros([2, z.shape[1]])
            for k in range(0, z.shape[1]):
                j = int(z[2][k])

                # if the landmark has never been seen before add it to the state vector
                if stateMeanBar[2 * j + 1][0] == 0:
                    landmark_pos = np.array([z[0, k] * math.cos(z[1, k] + stateMeanBar[2][0]),
                                             z[0, k] * math.sin(z[1, k] + stateMeanBar[2][0])]).reshape(2, 1)
                    stateMeanBar[2 * j + 1:2 * j + 3] = np.array([stateMeanBar[0][0] + landmark_pos[0][0],
                                                                  stateMeanBar[1][0] + landmark_pos[1][0]]).reshape(2,
                                                                                                                    1)
                    
                    continue

                # compute predicted range and bearing to the landmark
                delta = np.array([stateMeanBar[2 * j + 1][0] - stateMeanBar[0][0],
                                  stateMeanBar[2 * j + 2][0] - stateMeanBar[1][0]]).reshape(2, 1)
                q = np.matmul(delta.T, delta)[0][0]
                r = math.sqrt(q)  # predicted range to landmark

                # predicted bearing to landmark
                pred_bear = con_bear(math.atan2(delta[1][0], delta[0][0]) - stateMeanBar[2][0])

                # create predicted measurement
                zHat[:, k] = np.array([r, pred_bear])


                # calculate Jacobian of the measurement model
                h_t = np.array([[-delta[0][0] / r, -delta[1][0] / r, 0, delta[0][0] / r, delta[1][0] / r],
                                [delta[1][0] / q, -delta[0][0] / q, -1, -delta[1][0] / q, delta[0][0] / q]]).reshape(2,5)


                F_1 = np.vstack([np.identity(3), np.zeros([2, 3])])
                F_2 = np.vstack([np.zeros([3, 2]), np.identity(2)])
                F_xj = np.hstack([F_1, np.zeros([5, 2 * j - 2]), F_2, np.zeros([5, 2 * n_landmarks - 2 * j])])

                H_t = np.matmul(h_t, F_xj)

                # compute Kalman gain
                S = np.matmul(np.matmul(H_t, stateCovBar), H_t.T) + Q_t
                K = np.matmul(np.matmul(stateCovBar, H_t.T), np.linalg.inv(S))

                # incorporate new measurement into state mean and covariance
                stateMeanBar = stateMeanBar + np.matmul(K, (z[:2, k] - zHat[:, k]).reshape(2, 1))
                stateCovBar = np.matmul(np.identity(2 * n_landmarks + 3) - np.matmul(K, H_t), stateCovBar)

        # update state mean and covariance
        stateMean = stateMeanBar
        stateMean[2][0] = con_bear(stateMean[2][0])
        stateCov = stateCovBar

        # add new pose mean to estimated poses
        Robot_Est[i, :] = [t, stateMean[0][0], stateMean[1][0], stateMean[2][0]]

    # calculate land_loss: sum of squares of error in final landmark position
    land_loss = 0
    for i in range(n_landmarks):
        x_diff = stateMean[2 * i + 3][0] - landmark_groundtruth[i][1]
        y_diff = stateMean[2 * i + 4][0] - landmark_groundtruth[i][2]
        sq_err = math.sqrt(x_diff ** 2 + y_diff ** 2)
        land_loss = land_loss + sq_err
    land_loss = land_loss/n_landmarks
    print('average land_loss: ', land_loss)

    # computes euclidean loss between robot's estimated path and ground truth ignores bearing error
    path_loss = 0
    for i in range(start, robot_groundtruth.shape[0]):
        x_diff = robot_groundtruth[i][1] - Robot_Est[i, 1]
        y_diff = robot_groundtruth[i][2] - Robot_Est[i, 2]
        err = math.sqrt(x_diff ** 2 + y_diff ** 2)
        path_loss = path_loss + err
    path_loss = path_loss / len(range(start, robot_groundtruth.shape[0]))
    print('average path_loss: ', path_loss)

    return K, Robot_Est, land_loss, path_loss


if __name__ == '__main__':
    deltaT = .02
    alphas = np.array([.1, .01, .18, .08, 0, 0])  # robot-dependent motion noise parameters
    # measurement model noise parameters
    Q_t = np.array([[11.8, 0], [0, 0.18]])

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

    K, Robot_Est, land_loss, path_loss = main_ekf_slam(
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
