import numpy as np


def get_observations(robot_measurement, t, index, code_dict):
    # build vector of features observed at current time
    z = np.zeros([3, 1])
    while robot_measurement[index, 0] - t < 0.005 and index < robot_measurement.shape[0] - 1:
        # print(index)
        barcode = robot_measurement[index, 1]
        landmark_id = 0
        if barcode in code_dict:
            landmark_id = code_dict[barcode]
        else:
            print('key not found')
        if (landmark_id > 5) and (landmark_id < 21):
            range = robot_measurement[index, 2]
            bearing = robot_measurement[index, 3]
            if int(z[2][0]) == 0:
                z = np.array([range, bearing, landmark_id - 5]).reshape(3, 1)
            else:
                new_z = np.array([range, bearing, landmark_id - 5]).reshape(3, 1)
                z = np.hstack([z, new_z])

        index += 1

    return z, index
