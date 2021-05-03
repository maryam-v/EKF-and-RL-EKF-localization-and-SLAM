import numpy as np
import math
import random
from collections import deque
import tensorflow as tf



class Environment:
    def __init__(self,robot_groundtruth, landmark_groundtruth,
                 robot_measurement, robot_odometry, barcodes, codeDict, deltaT ):
        self.robot_groundtruth = robot_groundtruth 
        self.landmark_groundtruth = landmark_groundtruth
        self.n_landmarks = 15
        self.robot_measurement = robot_measurement
        self. robot_odometry = robot_odometry
        self.barcodes = barcodes
        self.codeDict = codeDict
        self.deltaT = 0.02
        self.episode_step = 0
        self.timestep = 599
        self.avg_error = 100
        self.kalman_filter_avg_path_loss = 1.0225804645624064
        self.Robot_Est = np.zeros([self.robot_groundtruth.shape[0], 4])
        self.last_measurement = np.zeros([3,1])
        self.measured = 0
        self.measurementIndex = 0
        
        self.min_v = np.min(self.robot_odometry[:,1])
        self.max_v = np.max(self.robot_odometry[:,1])
        self.min_w = np.min(self.robot_odometry[:,2])
        self.max_w = np.max(self.robot_odometry[:,2])
        
        self.observation_space = np.zeros([(2 * self.n_landmarks + 3)*(2 * self.n_landmarks + 4),])
        self.action_space = np.zeros([2,1])
        
        #self.last_landmark_ID = [11]
        
        self.alphas = np.array([.1, .01, .18, .08, 0, 0])
        self.Q_t = np.array([[11.8, 0], [0, 0.18]])
        
        self.last_stateCov = np.zeros([3+2 * self.n_landmarks,3+2 * self.n_landmarks])
        
        self.F_x = np.hstack([np.identity(3), np.zeros([3, 2 * self.n_landmarks])])

#        self.min_robot_groundtruth_x = min(self.robot_groundtruth[599:,1])
#        self.range_robot_groundtruth_x = max(self.robot_groundtruth[599:,1]) - min(self.robot_groundtruth[599:,1])
#        self.min_robot_groundtruth_y = min(self.robot_groundtruth[599:,2])
#        self.range_robot_groundtruth_y = max(self.robot_groundtruth[599:,2]) - min(self.robot_groundtruth[599:,2])
#        self.min_robot_groundtruth_theta = min(self.robot_groundtruth[599:,3])
#        self.range_robot_groundtruth_theta = max(self.robot_groundtruth[599:,3]) - min(self.robot_groundtruth[599:,3])
#        
#        self.min_robot_measurement_r = min(self.robot_measurement[:,2])
#        self.range_robot_measurement_r = max(self.robot_measurement[:,2]) - min(self.robot_measurement[:,2])
#        self.min_robot_measurement_phi = min(self.robot_measurement[:,3])
#        self.range_robot_measurement_phi = max(self.robot_measurement[:,3]) - min(self.robot_measurement[:,3])
#        
        
    def get_observations(self, index):
        # build vector of features observed at current time
        z = np.zeros([3, 1])
        while self.robot_measurement[index, 0] - self.t < 0.005 and index < self.robot_measurement.shape[0] - 1:
            self.barcode = self.robot_measurement[index, 1]
            landmark_id = 0
            if self.barcode in self.codeDict:
                landmark_id = self.codeDict[self.barcode]
            else:
                print('key not found')
            if (landmark_id > 5) and (landmark_id < 21):
                range = self.robot_measurement[index, 2]
                bearing = self.robot_measurement[index, 3]
                if int(z[2][0]) == 0:
                    z = np.array([range, bearing, landmark_id - 5]).reshape(3, 1)
                else:
                    new_z = np.array([range, bearing, landmark_id - 5]).reshape(3, 1)
                    z = np.hstack([z, new_z])
    
            index += 1
    
        return z, index
    

    def process_state(self):
        state = np.zeros([2 * self.n_landmarks + 3, 2 * self.n_landmarks + 4])
        state[:,0] = self.stateMean.reshape(2 * self.n_landmarks + 3,)
        state[:,1:] = self.stateCov.reshape(2 * self.n_landmarks + 3, 2 * self.n_landmarks + 3)
        state = state.reshape([(2 * self.n_landmarks + 3)*(2 * self.n_landmarks + 4),])
        
        
        return state
        
        
    def reset(self,start):        
        self.episode_step = 0
        if start == 599:
            self.stateMean = np.array([self.robot_groundtruth[start, 1],
                                      self.robot_groundtruth[start, 2],
                                      self.robot_groundtruth[start, 3]]).reshape(3, 1)
    
            self.F_x = np.hstack([np.identity(3), np.zeros([3, 2 * self.n_landmarks])])
            self.stateMean = np.matmul(self.F_x.T, self.stateMean)
            
            # initialize diagonal of pose covariance with small, nonzero values since we have a good estimate of the starting pose
            self.stateCov = np.zeros([2 * self.n_landmarks + 3, 2 * self.n_landmarks + 3])
            self.stateCov[:3, :3] = 0.001
        
            # initialize landmark variances to large values since we have no prior information on the locations of the landmarks
            for i in range(3, self.n_landmarks * 2 + 3):
                self.stateCov[i, i] = 10
    
    

        self.t = self.robot_groundtruth[start,0]    
        #z, self.measurementIndex = self.get_observations(self.measurementIndex)
        observation = self.process_state()
        
        return observation
    
    
    def con_bear(self,old_bear):
        while old_bear < -math.pi:
            old_bear = old_bear + 2 * math.pi
        while old_bear > math.pi:
            old_bear = old_bear - 2 * math.pi
        new_bear = old_bear
        
        return new_bear
    
    
    def path_loss_steps(self,step,num_step):
        path_loss = 0
        for i in range(step-num_step+1, step+1):
            x_diff = self.robot_groundtruth[i][1] - self.Robot_Est[i, 1]
            y_diff = self.robot_groundtruth[i][2] - self.Robot_Est[i, 2]
            err = math.sqrt(x_diff ** 2 + y_diff ** 2)
            path_loss = path_loss + err
            self.avg_loss = path_loss/num_step
            
        return self.avg_loss
    
    
    
    def step(self, action, step):
        self.t = self.robot_groundtruth[step,0]
        done = False
        self.episode_step += 1
        self.timestep += 1
        
        v = action.numpy()[0]
        w = action.numpy()[1]
        u_t = np.array([v,w])
        print(u_t)
        
        rot = self.deltaT * u_t[1]
        halfRot = rot / 2;
        trans = u_t[0] * self.deltaT;
        theta = self.stateMean[2][0]
            
        poseUpdate = np.array([trans * math.cos(theta + halfRot),
                               trans * math.sin(theta + halfRot),
                               rot]).reshape(3, 1)
            

        # updated state mean and constrain bearing to +/- pi
        stateMeanBar = self.stateMean + np.matmul(self.F_x.T, poseUpdate)
        stateMeanBar[2][0] = self.con_bear(stateMeanBar[2][0])
        
        
        # calculate the movement Jacobian
        g_t = np.array([[0, 0, -trans * math.sin(theta + halfRot)],
                        [0, 0, trans * math.cos(theta + halfRot)],
                        [0, 0, 0]])
    
        G_t = np.identity(2 * self.n_landmarks + 3) + np.matmul(np.matmul(self.F_x.T, g_t), self.F_x)

        # calculate motion covariance in control space
        M_t = np.array([[(self.alphas[0] * abs(u_t[0]) + self.alphas[1] * abs(u_t[1])) ** 2, 0],
                        [0, (self.alphas[2] * abs(u_t[0]) + self.alphas[3] * abs(u_t[1])) ** 2]])
        # M_t = np.array([[alphas[0] * abs(u_t[0][0])**2 + alphas[1] * abs(u_t[1][0])**2,0],
        #                [0,alphas[2] * abs(u_t[0][0])**2 + alphas[3] * abs(u_t[1][0])**2]])

        # calculate Jacobian to transform motion covariance to state space
        V_t = np.array([[math.cos(theta + halfRot), -0.5 * math.sin(theta + halfRot)],
                        [math.sin(theta + halfRot), 0.5 * math.cos(theta + halfRot)],
                        [0, 1]])


        # update state covariance
        R_t = np.matmul(np.matmul(V_t, M_t), V_t.T)
        stateCovBar = np.matmul(np.matmul(G_t, self.stateCov), G_t.T) + np.matmul(np.matmul(self.F_x.T, R_t), self.F_x)
      
        z, self.measurementIndex = self.get_observations(self.measurementIndex)
        
        #S = np.zeros([z.shape[1], 3, 3])
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
                pred_bear = self.con_bear(math.atan2(delta[1][0], delta[0][0]) - stateMeanBar[2][0])

                # create predicted measurement
                zHat[:, k] = np.array([r, pred_bear])
                                                
                # calculate Jacobian of the measurement model
                h_t = np.array([[-delta[0][0] / r, -delta[1][0] / r, 0, delta[0][0] / r, delta[1][0] / r],
                                [delta[1][0] / q, -delta[0][0] / q, -1, -delta[1][0] / q, delta[0][0] / q]]).reshape(2,5)


                F_1 = np.vstack([np.identity(3), np.zeros([2, 3])])
                F_2 = np.vstack([np.zeros([3, 2]), np.identity(2)])
                F_xj = np.hstack([F_1, np.zeros([5, 2 * j - 2]), F_2, np.zeros([5, 2 * self.n_landmarks - 2 * j])])

                H_t = np.matmul(h_t, F_xj)

                # compute Kalman gain
                S = np.matmul(np.matmul(H_t, stateCovBar), H_t.T) + self.Q_t
                K = np.matmul(np.matmul(stateCovBar, H_t.T), np.linalg.inv(S))

                # incorporate new measurement into state mean and covariance
                stateMeanBar = stateMeanBar + np.matmul(K, (z[:2, k] - zHat[:, k]).reshape(2, 1))
                stateCovBar = np.matmul(np.identity(2 * self.n_landmarks + 3) - np.matmul(K, H_t), stateCovBar)
                                           
        # update state mean and covariance
        self.stateMean = stateMeanBar
        self.stateMean[2][0] = self.con_bear(self.stateMean[2][0])
        self.stateCov = stateCovBar
        
        new_observation = self.process_state()

        # add new pose mean to estimated poses
        self.Robot_Est[step, :] = [self.t, self.stateMean[0][0], self.stateMean[1][0], self.stateMean[2][0]]
        
        
        
        uncertainty_diff = -np.linalg.det(self.stateCov)+np.linalg.det(self.last_stateCov)
        reward_step = uncertainty_diff
        
        if self.episode_step>=10:
            done=True

        self.last_stateCov = self.stateCov          

        return new_observation, reward_step, done
    
    
    def choose_random_action(self):
        rand_action = np.zeros([1,2]).reshape(2,)
        rand_action[0] = random.uniform(self.min_v, self.max_v)
        rand_action[1] = random.uniform(self.min_w, self.max_w)
        action = tf.convert_to_tensor(rand_action)
        return action
    
