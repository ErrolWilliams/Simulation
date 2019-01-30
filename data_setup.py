import os
import numpy as np
from view import show


def split_data(data_file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_src = os.path.join(dir_path, data_file)

    data = np.load(data_src)

    num_sensors = len(data[0][0]) - 1

    new_data = np.ndarray(shape=(1000, 20, 50, 50, 2), dtype=np.uint8)

    for i in range(num_sensors):
        train_data_dest = os.path.join(dir_path, 'sensor_data/sensor{0}_data'.format(i))
        if not os.path.exists(train_data_dest):
            os.makedirs(train_data_dest)

    env_data_dest = os.path.join(dir_path, 'sensor_data/env_data')
    if not os.path.exists(env_data_dest):
        os.makedirs(env_data_dest)

    for i, sequence in enumerate(data):  # for each sequence
        env_states = []
        save_envs = False
        for j in range(num_sensors):
            for k in range(20):
                if len(env_states) < 20:
                    env_states.append(sequence[k][0])
                # show(env_states[k])
                new_data[i, k, :, :, 0] = sequence[k][j+1][1]  # sensor i occupancy grid
                # show(new_data[i, k, :, :, 0])
                new_data[i, k, :, :, 1] = sequence[k][j+1][0]  # sensor i visibility grid
                # show(new_data[i, k, :, :, 1])
            if not save_envs:
                np.save(env_data_dest + '/{0}.npy'.format(i+9000), env_states)
            np.save('sensor_data/sensor{0}_data/{1}.npy'.format(j, i+9000), new_data[i, :, :, :, :])


split_data('training_data200000.npy')
