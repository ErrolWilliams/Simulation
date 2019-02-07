import pandas as pd
import numpy as np
import time
import os

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
from view import show_predictions

K.set_image_data_format("channels_last")

# setup('file')
# exit(0)

# Datasets indexes. Folder "data" should contain a different file for each sequence of 20 frames on a file "1.npy", "2.npy"...."10000.npy"

# val_index = np.load('testData/0.npy')
# tr_index = np.load('testData/5.npy')
val_index = np.arange(4000, 5000)
tr_index = np.arange(4000)
mode_test = False
test_multi = False
reload_model = False
learning_rate = 0.01
model_group = 'models/test_models/'
model_folder = 'test_multi/100-15.68.hdf5'
path_model = model_group + model_folder
if not os.path.exists(path_model):
    os.makedirs(path_model)
initial_epoch = 0
# Parameters

# Input dimensions, parameters for training
params = {'dim': (20, 50, 50),
          'nbatch': 10,
          'n_channels': 2,
          'shuffle': True,
          'load_path': ['sensor_data/top_sensor_data/', 'sensor_data/left_sensor_data/']}

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
csvlogName = path_model + '.csv'
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
                        epochs=100, verbose=1,
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
    input = np.ndarray(shape=(1, 20, 50, 50, 2))
    env = np.ndarray(shape=(20, 50, 50))
    if not test_multi:
        input[:, :, :, :, :] = np.load('sensor_data/right_sensor_data/0.npy')  # vis and occ grids
        env[:, :, :] = np.load('sensor_data/env_data/0.npy')  # env grids
    else:
        # right_data = np.load('sensor_data/right_sensor_data/0.npy')  # 20 vis and occ grids for right sensor
        top_data = np.load('sensor_data/top_sensor_data/0.npy')  # 20 vis and occ grids for top sensor
        # left_data = np.load('sensor_data/left_sensor_data/0.npy')  # 20 vis and occ grids for left sensor
        bottom_data = np.load('sensor_data/bottom_sensor_data/0.npy')  # 20 vis and occ grids for bottomsensor

        input_data = np.bitwise_or(top_data, bottom_data)
        # input_data = np.bitwise_or(data1, data3)
        # input_data = np.bitwise_or(input_data, data4)
        # input_data = input_data.astype(np.int, copy=False)
        input[:, :, :, :, :] = input_data
        env[:, :, :] = np.load('sensor_data/env_data/0.npy')  # 20 env grids
    output = model.predict(input)
    input = input[:, :10, :, :, :]
    show_predictions(input, output, env)
    exit(0)
