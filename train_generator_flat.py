import pandas as pd
import numpy as np
import time
import os
from view import show
import csv

#############  KERAS OBJECTS
# from __future__ import print_function
import keras

from keras.models import Model
from keras.layers import Input, Conv2D, TimeDistributed, ConvLSTM2D, Dense, Cropping3D
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf
from my_classes import DataGenerator_flat, DataGenerator_flat_multi
from models import model_flat
from ops import *
from view import show_predictions, get_stats

K.set_image_data_format("channels_last")

# setup('file')
# exit(0)

# Datasets indexes. Folder "data" should contain a different file for each sequence of 20 frames on a file "1.npy", "2.npy"...."10000.npy"

# val_index = np.load('testData/0.npy')
# tr_index = np.load('testData/5.npy')
val_index = np.arange(4000, 5000)
tr_index = np.arange(4000)
mode_test = True
test_multi = False
reload_model = False
learning_rate = 0.01
width = 25
height = 25
model_group = 'models/test_models/'
model_folder = 'bottom61/'
data_folder = 'sensor_data_61'
path_model = model_group + model_folder
if not os.path.exists(path_model):
    os.makedirs(path_model)
initial_epoch = 0
# Parameters

# Input dimensions, parameters for training
params = {'dim': (20, width, height),
          'nbatch': 10,
          'n_channels': 2,
          'shuffle': True,
          'load_path': [data_folder + '/bottom_sensor_data/']}

# ['sensor_data/left_sensor_data/', 'sensor_data/right_sensor_data/']
input_shape = (*params['dim'], params['n_channels'])

# For testing on 100 samples on CPU
if mode_test:
    partition = {'train': np.arange(100), 'validation': np.arange(100, 200)}
else:
    partition = {'train': tr_index, 'validation': val_index}

training_generator = DataGenerator_flat_multi(partition['train'], **params)
validation_generator = DataGenerator_flat_multi(partition['validation'], **params)

# Define model
model = model_flat(input_shape=input_shape, learning_rate=learning_rate, training=True)

# Write a csv file with loss function after each epoch
csvlogName = path_model + 'log.csv'
fp = path_model + '{epoch:02d}-{loss:.2f}.hdf5'

csv_logger = CSVLogger(csvlogName, append=True)

# Reduce learning rate if loss is not decreasing
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=10, epsilon=0.000001, min_lr=0.000001)

# Save last model weights
checkpointer = ModelCheckpoint(filepath=fp, monitor='loss', verbose=0,
                               save_best_only=False,
                               save_weights_only=False,
                               mode='auto',
                               period=1)
# Load pretrained model
if reload_model:
    model.load_weights(path_model)

if not mode_test:
    print('Started training')
    t0 = time.time()

    model.fit_generator(generator=training_generator,
                        epochs=50, verbose=1,
                        validation_data=validation_generator,
                        # use_multiprocessing=True,
                        initial_epoch=initial_epoch,
                        callbacks=[checkpointer,
                                   reduce_lr,
                                   csv_logger])

    tf = time.time()
    time_file = np.array([t0, tf, tf - t0])
    np.save(path_model + 'time.npy', time_file)

if mode_test:
    sample_num = 1
    model_files = os.listdir(path_model)
    model_files = [x for x in model_files if '.hdf5' in x]
    model_files = model_files[:30]
    ball_error_means = []
    empty_error_means = []
    for file in model_files:
        print(file)
        model.load_weights(path_model + file)
        ball_errors_total = 0
        empty_errors_total = 0
        for i in range(sample_num):
            input = np.ndarray(shape=(1, 20, width, height, 2))
            env = np.ndarray(shape=(20, width, height))
            right_data = np.load(data_folder + '/right_sensor_data/{0}.npy'.format(i))  # 20 vis and occ grids for right sensor
            top_data = np.load(data_folder + '/top_sensor_data/{0}.npy'.format(i))  # 20 vis and occ grids for top sensor
            left_data = np.load(data_folder + '/left_sensor_data/{0}.npy'.format(i))  # 20 vis and occ grids for left sensor
            bottom_data = np.load(data_folder + '/bottom_sensor_data/{0}.npy'.format(i))  # 20 vis and occ grids for bottomsensor

            input_data = bottom_data
            #input_data = np.bitwise_or(input_data, bottom_data)
            #input_data = np.bitwise_or(input_data, top_data)
            #input_data = np.bitwise_or(input_data, right_data)
            #input_data = input_data.astype(np.int, copy=False)
            input[:, :, :, :, :] = input_data
            env[:, :, :] = np.load(data_folder + '/env_data/{0}.npy'.format(i))  # 20 env grids
            output = model.predict(input)
            input = input[:, :10, :, :, :]
            show_predictions(input, output, env, width, height)
            ball_error_pct, empty_error_pct = get_stats(output, env, width, height)
            ball_errors_total += ball_error_pct
            empty_errors_total += empty_error_pct
        ball_error_means.append(ball_errors_total/sample_num)
        empty_error_means.append(empty_errors_total/sample_num)

    """
    rows = zip(ball_error_means, empty_error_means)
    with open(path_model + 'l_r_errors-2.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in rows:
            wr.writerow(row)
            
    env = np.load('sensor_data_33/env_data/0.npy')
    top = np.load('sensor_data_33/right_sensor_data/0.npy')
    show(env[0])
    show(top[15, :, :, 0])
    """
