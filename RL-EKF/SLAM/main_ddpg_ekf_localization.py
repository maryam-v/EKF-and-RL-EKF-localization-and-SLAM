import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve, path_loss
from environment import Environment
from prepare_data import load_mrclam_dataset
from prepare_data import sample_mrclam_dataset
from animate_mrclam_dataset import animate_mrclam_dataset


if __name__ == '__main__':
    deltaT = .02
    barcodes, landmark_groundtruth, n_landmarks, \
    robot_groundtruth, robot_odometry, robot_measurement = \
        load_mrclam_dataset()

    robot_groundtruth, robot_odometry, robot_measurement, timesteps = \
        sample_mrclam_dataset(
            robot_groundtruth,
            robot_odometry,
            robot_measurement,
            sample_time=deltaT
        )

    # set up map between barcodes and landmark IDs
    codeDict = dict()
    j = 0
    for i in barcodes[:, 1]:
        codeDict[i] = barcodes[j, 0]
        j += 1

    env = Environment(robot_groundtruth, landmark_groundtruth,
                      robot_measurement, robot_odometry, barcodes, codeDict, deltaT )
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])


    figure_file = 'plots/DDPG-EKF-Reward.png'
    best_score = 0
    score_history = []
            
        
    # training
    

    step = 599
    start_train_step = 599
    end_training_step = 75000  
    episode = 0
    evaluate = False

    while step < end_training_step:
        observation = env.reset(step)
        done = False
        score = 0
        while not done and step < end_training_step:
            action = agent.choose_action(observation,evaluate)
            observation_, reward, done = env.step(action,step)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            step+=1

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

#        if avg_score > best_score:
#            best_score = avg_score
#            agent.save_models()

        episode += 1
        print('episode ', episode, 'score %.4f' % score, 'step', step)
        
    avg_loss = path_loss(start_train_step, end_training_step, robot_groundtruth, Robot_Est=env.Robot_Est)
    print('average path loss for training: ', avg_loss)

#    if not load_checkpoint:
    x = [i+1 for i in range(episode)]
    plot_learning_curve(x, score_history, figure_file)
    
    # Evaluating
#    evaluate = True
#    #agent.load_models()
#    step = 60000
#    start_evaluating_step = 60000
#    end_evaluating_step = 75000  
#    episode = 0
#    while step < end_evaluating_step:
#        observation = env.reset(step)
#        done = False
#        score = 0
#        while not done and step < end_evaluating_step:
#            action = agent.choose_action(observation, evaluate)
#            observation_, reward, done = env.step(action,step)
#            score += reward
#            agent.remember(observation, action, reward, observation_, done)
#            observation = observation_
#
#            step+=1
#
#
#        episode += 1
#        print('episode ', episode, 'score %.4f' % score, 'step', step)
#        
#        
#    avg_loss = path_loss(start_evaluating_step, end_evaluating_step, robot_groundtruth=robot_groundtruth, Robot_Est=env.Robot_Est)
#    print('average path loss for evaluation: ', avg_loss)
        
    
#    animate_mrclam_dataset(
#        robot_groundtruth,    
#        env.Robot_Est,
#        landmark_groundtruth,
#        timesteps,
#        robot_measurement
#)    
    