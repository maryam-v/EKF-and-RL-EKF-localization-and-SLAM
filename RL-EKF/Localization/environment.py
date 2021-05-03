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
        
        self.observation_space = np.zeros([16,])
        self.action_space = np.zeros([2,1])
        
        self.last_landmark_ID = [11]
        
        self.alphas = np.array([.2, .03, .09, .08])
        
        self.last_posCov = np.zeros([3,3])

        self.min_robot_groundtruth_x = min(self.robot_groundtruth[599:,1])
        self.range_robot_groundtruth_x = max(self.robot_groundtruth[599:,1]) - min(self.robot_groundtruth[599:,1])
        self.min_robot_groundtruth_y = min(self.robot_groundtruth[599:,2])
        self.range_robot_groundtruth_y = max(self.robot_groundtruth[599:,2]) - min(self.robot_groundtruth[599:,2])
        self.min_robot_groundtruth_theta = min(self.robot_groundtruth[599:,3])
        self.range_robot_groundtruth_theta = max(self.robot_groundtruth[599:,3]) - min(self.robot_groundtruth[599:,3])
        
        self.min_robot_measurement_r = min(self.robot_measurement[:,2])
        self.range_robot_measurement_r = max(self.robot_measurement[:,2]) - min(self.robot_measurement[:,2])
        self.min_robot_measurement_phi = min(self.robot_measurement[:,3])
        self.range_robot_measurement_phi = max(self.robot_measurement[:,3]) - min(self.robot_measurement[:,3])
        
        sigma_range = 2
        sigma_bearing = 3
        sigma_id = 1
        self.Q_t = np.array([[sigma_range ** 2, 0, 0], [0, sigma_bearing ** 2, 0], [0, 0, sigma_id ** 2]])
        
        
        self.action_list = deque(maxlen=4)
        self.action_list.append(0)
        self.action_list.append(0)
        self.action_list.append(0)
        self.action_list.append(0)
        
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
        state = np.zeros([16,])
        state[0:3,] = self.poseMean.reshape(3,)
        state[3:12] = self.poseCov.reshape(9,)
        state[12:16] = self.action_list
        
#        if measurement[2][0]>1:
#            landmarks_id = measurement[2,:]
#            for i in range(measurement.shape[1]):
#                land_num = int(landmarks_id[i])
#                #state[2*land_num+1:2*land_num+3,] = measurement[0:2,i].reshape([2,])
#                state[2*land_num+1,] = (measurement[0,i].reshape([1,])+self.min_robot_measurement_r)/self.range_robot_measurement_r
#                state[2*land_num+2,] = (measurement[1,i].reshape([1,])+self.min_robot_measurement_phi)/self.range_robot_measurement_phi
                
#        state[0] = (state[0] + self.min_robot_groundtruth_x)/self.range_robot_groundtruth_x
#        state[1] = (state[1] + self.min_robot_groundtruth_y)/self.range_robot_groundtruth_y
#        state[2] = (state[2] + self.min_robot_groundtruth_theta)/self.range_robot_groundtruth_theta
            
        return state
        
        
    def reset(self,start):
        self.episode_step = 0
        if start == 599:
            self.poseMean = np.array([self.robot_groundtruth[start, 1],
                                      self.robot_groundtruth[start, 2],
                                      self.robot_groundtruth[start, 3]]).reshape(3, 1)
    
    
            self.poseCov = np.array([[0.01, 0.01, 0.01],
                                     [0.01, 0.01, 0.01],
                                     [0.01, 0.01, 0.01]])


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
        self.action_list.append(v)
        self.action_list.append(w)
        
        rot = self.deltaT * w
        halfRot = rot / 2;
        trans = v * self.deltaT;
        theta = self.poseMean[2][0]
        
        # calculate the movement Jacobian
        G_t = np.array([[1, 0, -trans * math.sin(theta + halfRot)],
                        [0, 1, trans * math.cos(theta + halfRot)],
                        [0, 0, 1]])

        # calculate motion covariance in control space
        M_t = np.array([[(self.alphas[0] * abs(v) + self.alphas[1] * abs(w)) ** 2, 0],
                        [0, (self.alphas[2] * abs(v) + self.alphas[3] * abs(w)) ** 2]])
        # M_t = np.array([[alphas[0] * abs(u_t[0][0])**2 + alphas[1] * abs(u_t[1][0])**2,0],
        #                [0,alphas[2] * abs(u_t[0][0])**2 + alphas[3] * abs(u_t[1][0])**2]])

        # calculate Jacobian to transform motion covariance to state space
        V_t = np.array([[math.cos(theta + halfRot), -0.5 * math.sin(theta + halfRot)],
                        [math.sin(theta + halfRot), 0.5 * math.cos(theta + halfRot)],
                        [0, 1]])
    
        poseUpdate = np.array([trans * math.cos(theta + halfRot),
                               trans * math.sin(theta + halfRot),
                               rot]).reshape(3, 1)
    
        # calculate estimated pose mean
        poseMeanBar = self.poseMean + poseUpdate

        # get measurements for the current timestep
        poseCovBar = np.matmul(np.matmul(G_t, self.poseCov), G_t.T) + np.matmul(np.matmul(V_t, M_t), V_t.T)

        #self.poseMean += poseUpdate
        #self.poseMean[2][0] = self.con_bear(self.poseMean[2][0])
        
        #self.Robot_Est[step, :] = [self.t, self.poseMean[0][0], self.poseMean[1][0], self.poseMean[2][0]]

        
        
      
        z, self.measurementIndex = self.get_observations(self.measurementIndex)
        

        S = np.zeros([z.shape[1], 3, 3])
        zHat = np.zeros([3, z.shape[1]])
        
#        if step < 636:
#            z = np.array([2.148, 0.025, 11]).reshape(3, 1)
#            self.last_measurement = z
#            self.last_landmark_ID = [11]
#            zHat = np.zeros([3, self.last_measurement.shape[1]])
#            for k in range(0, z.shape[1]):
#                j = z[2][k]
#    
#                # get coordinates of the measured landmark
#                m = self.landmark_groundtruth[int(j) - 1, 1:3]
#    
#                # compute the expected measurement
#                xDist = m[0] - self.poseMean[0][0]
#                yDist = m[1] - self.poseMean[1][0]
#                q = xDist ** 2 + yDist ** 2
#    
#                # constrains expected bearing to between 0 and 2*pi
#                pred_bear = self.con_bear(math.atan2(yDist, xDist) - self.poseMean[2][0])
#                zHat[:, k] = [math.sqrt(q), pred_bear, j]
#        
#        else:
        
        # if any measurements are available
        if z[2][0] > 1:
            #self.last_landmark_ID = []
            #self.last_measurement = z
            for k in range(0, z.shape[1]):
                j = z[2][k]
    
                # get coordinates of the measured landmark
                m = self.landmark_groundtruth[int(j) - 1, 1:3]
    
                # compute the expected measurement
                xDist = m[0] - self.poseMean[0][0]
                yDist = m[1] - self.poseMean[1][0]
                q = xDist ** 2 + yDist ** 2
    
                # constrains expected bearing to between 0 and 2*pi
                pred_bear = self.con_bear(math.atan2(yDist, xDist) - self.poseMean[2][0])
                zHat[:, k] = [math.sqrt(q), pred_bear, j]
                
#                self.measured = 1
#                self.last_measurement_step = step
#                self.last_landmark_ID.append(j) 
                
                
                # calculate Jacobian of the measurement model
                H = np.array([[-1 * (xDist / math.sqrt(q)), -1 * (yDist / math.sqrt(q)), 0],
                              [yDist / q, -xDist / q, -1],
                              [0, 0, 0]])

                # compute S
                S[k, :, :] = np.matmul(np.matmul(H, poseCovBar), H.T) + self.Q_t

                # compute Kalman gain
                K = np.matmul(np.matmul(self.poseCov, H.T), np.linalg.inv(S[k, :, :]))

                # update pose mean and covariance estimates
                poseMeanBar = poseMeanBar + np.matmul(K, (z[:, k] - zHat[:, k])).reshape(3, 1)
                poseCovBar = np.matmul(np.identity(3) - np.matmul(K, H), poseCovBar)
                    
                    
#            else:
#                zHat = np.zeros([3, self.last_measurement.shape[1]])
#                for k in range(len(self.last_landmark_ID)):
#                    j = self.last_landmark_ID[k]
#    
#                    m = self.landmark_groundtruth[int(j) - 1, 1:3]
#        
#                    # compute the expected measurement
#                    xDist = m[0] - self.poseMean[0][0]
#                    yDist = m[1] - self.poseMean[1][0]
#                    q = xDist ** 2 + yDist ** 2
#        
#                    # constrains expected bearing to between 0 and 2*pi
#                    pred_bear = self.con_bear(math.atan2(yDist, xDist) - self.poseMean[2][0])
#                    zHat[:, k] = [math.sqrt(q), pred_bear, j]
            
                
                
        # update pose mean and covariance constrains heading to between 0 and 2*pi
        self.poseMean = poseMeanBar
        self.poseMean[2][0] = self.con_bear(self.poseMean[2][0])
        self.poseCov = poseCovBar
        
        new_observation = self.process_state()

        # add pose mean to estimated position vector
        self.Robot_Est[step, :] = [self.t, self.poseMean[0][0], self.poseMean[1][0], self.poseMean[2][0]]
        
        
        
        uncertainty_diff = -np.linalg.det(self.poseCov)+np.linalg.det(self.last_posCov)
        reward_step = uncertainty_diff
        #print('u')
        
        if self.episode_step>=10:
            done=True

#        error = 0
#        for k in range(0, self.last_measurement.shape[1]):
#            x_diff = self.last_measurement[0,k]*math.cos(self.last_measurement[1,k])-zHat[0,k]*math.cos(zHat[1,k])
#            y_diff = self.last_measurement[0,k]*math.sin(self.last_measurement[1,k])-zHat[0,k]*math.sin(zHat[1,k])
#            error += math.sqrt(x_diff**2 +y_diff**2)
#        error /= self.last_measurement.shape[1]
#        error /=20
        
#        if self.episode_step >= 100:
#            reward_step = max(1-error,0) 
#            done = True
#        
#        num_step = 10
#        if self.timestep-599>=num_step:
#            self.avg_loss = self.path_loss_steps(step,num_step)
#            if self.avg_loss < 1.2* self.kalman_filter_avg_path_loss:
#                done = True
#                reward_step = 100
#        if not done:
#            reward_step = max(1-error,0)  

        self.last_posCov = self.poseCov          

        return new_observation, reward_step, done
    
    
    def choose_random_action(self):
        rand_action = np.zeros([1,2]).reshape(2,)
        rand_action[0] = random.uniform(self.min_v, self.max_v)
        rand_action[1] = random.uniform(self.min_w, self.max_w)
        action = tf.convert_to_tensor(rand_action)
        return action
    
