import numpy as np
import math
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def con_bear(old_bear):
    while old_bear < -math.pi:
        old_bear = old_bear + 2 * math.pi
    while old_bear > math.pi:
        old_bear = old_bear - 2 * math.pi
    new_bear = old_bear

    return new_bear


def path_loss(start, end, robot_groundtruth, Robot_Est):
    # computes euclidean loss between robot's estimated path and ground truth ignores bearing error
    path_loss = 0
    for i in range(start, end):
        x_diff = robot_groundtruth[i][1] - Robot_Est[i, 1]
        y_diff = robot_groundtruth[i][2] - Robot_Est[i, 2]
        err = math.sqrt(x_diff ** 2 + y_diff ** 2)
        path_loss = path_loss + err
        avg_loss = path_loss/len(range(start, end))
    return avg_loss