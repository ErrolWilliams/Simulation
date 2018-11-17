import os
import numpy as np


def split_data(data_file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_src = os.path.join(dir_path, data_file)
    data_dest = os.path.join(dir_path, 'data')
    if not os.path.exists(data_dest):
        os.makedirs(data_dest)

    data = np.load(data_src)

    num_sensors = len(data[0][0]) - 1

    new_data = np.ndarray(shape=(1000, 20, 50, 50, 2), dtype=np.uint8)

    for i, sequence in enumerate(data):  # for each sequence
        for j in range(20):
            new_data[i, j, :, :, 0] = sequence[j][1][1]  # sensor 0 occupancy grid
            new_data[i, j, :, :, 1] = sequence[j][1][0]  # sensor 0 visibility grid

        np.save('data/{0}.npy'.format(i), new_data[i, :, :, :, :])


split_data('training_data20000.npy')
