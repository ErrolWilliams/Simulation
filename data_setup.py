import os
import numpy as np
from shapely.geometry import box, Point, LineString
from shapely.ops import cascaded_union, linemerge
from view import show
from simulation import Sensor

width = 25
height = 25

def split_data(data_src, data_dst_dir, save_num):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_src = os.path.join(dir_path, data_src)

    data = np.load(data_src)

    num_sensors = len(data[0][0]) - 1

    new_data = np.ndarray(shape=(1000, 20, width, height, 2), dtype=np.uint8)

    for i in range(num_sensors):
        train_data_dest = os.path.join(dir_path, '{0}/sensor{1}_data'.format(data_dst_dir, i))
        if not os.path.exists(train_data_dest):
            os.makedirs(train_data_dest)

    env_data_dest = os.path.join(dir_path, '{0}/env_data'.format(data_dst_dir))
    if not os.path.exists(env_data_dest):
        os.makedirs(env_data_dest)

    for i, sequence in enumerate(data):  # for each sequence
        env_states = []
        save_envs = False
        for j in range(num_sensors):  # for each sensor
            for k, step in enumerate(sequence):  # split sequence data
                env_states.append(step[0])
                show(step[0])
                vis = step[j + 1][0]
                occ = step[j + 1][1]
                new_data[i, k, :, :, 0] = occ  # sensor i occupancy grid
                show(new_data[i, k, :, :, 0])
                new_data[i, k, :, :, 1] = vis  # sensor i visibility grid
                show(new_data[i, k, :, :, 1])
            if not save_envs:
                np.save(env_data_dest + '/{0}.npy'.format(i + save_num), env_states)
            np.save('{0}/sensor{1}_data/{2}.npy'.format(data_dst_dir, j, i + save_num), new_data[i, :, :, :, :])


if __name__ == '__main__':
    data = np.load('sensor_data_61/env_data/0.npy')
    show(data[1])
    data_2 = np.load('sensor_data_61/sensor3_data/0.npy')
    show(data_2[1, :, :, 1])
    exit(0)
    for i in range(10):
        print(i+1)
        data_num = (i+1)*20000
        data_file = 'training_data61/training_data_{0}.npy'.format(data_num)
        split_data(data_file, 'sensor_data_61', i*1000)
