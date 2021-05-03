# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 23:09:20 2020

@author: Maryam
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def animate_mrclam_dataset(
        robot_groundtruth,
        robot_est,
        landmark_groundtruth,
        timesteps,
        robot_measurement
):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-6, 7)
    ax.set_xticklabels([k1 for k1 in np.arange(-10, 11, 2)])

    start_timestep = 0
    end_timestep = timesteps
    timesteps_per_frame = 50
    pause_time_between_frames = 0.01


    n_landmarks = landmark_groundtruth.shape[0]


    robot_colors = [1, 0, 0]
    estimated_robot_colors = [0, 0.67, 0]
    landmark_colors = [[0.3, 0.3, 0.3] for i in range(n_landmarks)]
    landmark_colors = np.array(landmark_colors)

    r_robot = 0.165/10
    d_robot = 20 * r_robot
    r_landmark = 0.055
    d_landmark = 2 * r_landmark

    # get positions for landmarks
    for i in range(n_landmarks):
        x = landmark_groundtruth[i, 1]
        y = landmark_groundtruth[i, 2]
        p1 = x - r_landmark
        p2 = y - r_landmark
        ax.add_patch(
            patches.Rectangle(
                (p1, p2), d_landmark, d_landmark,
                edgecolor=landmark_colors[i, :],
                facecolor=landmark_colors[i, :],
                fill=True,
                angle=45,
                linewidth=1
            )
        )

    # get initial positions for robot groundtruth
    x = robot_groundtruth[0, 1]
    y = robot_groundtruth[0, 2]
    z = robot_groundtruth[0, 3]
    x1 = d_robot * math.cos(z) + x
    y1 = d_robot * math.sin(z) + y
    p1 = x - r_robot
    p2 = y - r_robot
    rectangle_g = patches.Rectangle(
            (p1, p2), d_robot, d_robot,
            edgecolor=robot_colors,
            facecolor=robot_colors,
            fill=True,
            angle=0,
            linewidth=1
        )
    ax.add_patch(rectangle_g)

    ax.plot([x, x1], [y, y1], color='k', linewidth=1)

    # get initial positions for robot pose estimates
    x = robot_est[0, 1]
    y = robot_est[0, 2]
    z = robot_est[0, 3]
    x1 = d_robot * math.cos(z) + x
    y1 = d_robot * math.sin(z) + y
    p1 = x - r_robot
    p2 = y - r_robot
    rectangle_est = patches.Rectangle(
            (p1, p2), d_robot, d_robot,
            edgecolor=estimated_robot_colors,
            facecolor=estimated_robot_colors,
            fill=True,
            angle=0,
            linewidth=1
        )
    ax.add_patch(rectangle_est)

    ax.plot([x, x1], [y, y1], color='k', linewidth=1)

    for k in range(start_timestep, end_timestep):
        if (k + 1) % timesteps_per_frame == 0:
            rectangle_est.remove()
            rectangle_g.remove()

        x_g = robot_groundtruth[k, 1]
        y_g = robot_groundtruth[k, 2]
        z_g = robot_groundtruth[k, 3]

        x_est = robot_est[k, 1]
        y_est = robot_est[k, 2]
        z_est = robot_est[k, 3]

        if (k + 1) % timesteps_per_frame == 0:
            x1_g = d_robot * math.cos(z_g) + x_g
            y1_g = d_robot * math.sin(z_g) + y_g
            p1_g = x_g - r_robot
            p2_g = y_g - r_robot
            rectangle_g = patches.Ellipse(
                    (p1_g, p2_g), d_robot * 5 / 8, d_robot,
                    edgecolor=robot_colors,
                    facecolor=robot_colors,
                    fill=True,
                    angle=0,
                    linewidth=1
                )
            ax.add_patch(rectangle_g)
            ax.plot([x_g, x_g + (x1_g - x_g)* 0.5], [y_g, y_g + (y1_g - y_g) * 0.5], color=robot_colors, linewidth=1)

            x1_est = d_robot * math.cos(z_est) + x_est
            y1_est = d_robot * math.sin(z_est) + y_est
            p1_est = x_est - r_robot
            p2_est = y_est - r_robot
            rectangle_est = patches.Ellipse(
                    (p1_est, p2_est), d_robot * 5 / 8, d_robot,
                    edgecolor=estimated_robot_colors,
                    facecolor=estimated_robot_colors,
                    fill=True,
                    angle=0,
                    linewidth=1
                )
            ax.add_patch(rectangle_est)

            ax.plot([x_est, x_est + (x1_est-x_est)*0.5], [y_est, y_est + (y1_est-y_est)*0.5], color=estimated_robot_colors, linewidth=1)

            save_eps = True
            if save_eps and ((k + 1) % (20 * timesteps_per_frame) == 0):
                #images_path = str(os.getcwd()) + '/images/'
                plt.title('EKF-SLAM')
                #plt.savefig(f'{images_path}png_{int(k+1)}.png', dpi=100, format='png')

            plt.draw()
            plt.pause(pause_time_between_frames)

    plt.show()
