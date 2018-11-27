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
from keras.callbacks import CSVLogger, ReduceLROnPlateau

import tensorflow as tf
from my_classes import DataGenerator_flat
from models import model_flat
from ops import *

K.set_image_data_format("channels_last")

# setup('file')
# exit(0)

# Datasets indexes. Folder "data" should contain a different file for each sequence of 20 frames on a file "1.npy", "2.npy"...."10000.npy"

#val_index = np.load('testData/0.npy')
#tr_index = np.load('testData/5.npy')
val_index = np.arange(100, 200)
tr_index = np.arange(100)
mode_test = False
reload_model = False
learning_rate = 0.001
path_model = 'models/sensor_review_.98-0.04.hdf5'
initial_epoch = 0
# Parameters

# Input dimensions, parameters for training
params = {'dim': (20, 50, 50),
          'nbatch': 10,
          'n_channels': 2,
          'shuffle': True,
          'load_path': 'sensor1_data/'}

input_shape = (*params['dim'], params['n_channels'])

# For testing on 100 samples on CPU
if mode_test:
    partition = {'train': np.arange(100), 'validation': np.arange(100, 200)}
else:
    partition = {'train': tr_index, 'validation': val_index}

training_generator = DataGenerator_flat(partition['train'], **params)
validation_generator = DataGenerator_flat(partition['validation'], **params)

# Define model
model = model_flat(input_shape=input_shape, learning_rate=learning_rate, training=True)

# Write a csv file with loss function after each epoch
csvlogName = 'models/sensor_review_' + str(initial_epoch) + '.csv'
fp = 'models/sensor_review_.{epoch:02d}-{loss:.2f}.hdf5'

csv_logger = CSVLogger(csvlogName)

# Reduce learning rate if loss is not decreasing
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=10, epsilon=0.000001, min_lr=0.000001)

# Save last model weights
checkpointer = keras.callbacks.ModelCheckpoint(filepath=fp, monitor='loss', verbose=0,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto',
                                               period=1)
# Load pretrained model
if reload_model:
    model.load_weights(path_model)

print('Started training')
t0 = time.time()

if not mode_test:
    model.fit_generator(generator=training_generator,
                        epochs=100, verbose=1,
                        validation_data=validation_generator,
                        # use_multiprocessing=True,
                        initial_epoch=initial_epoch,
                        callbacks=[checkpointer,
                                   reduce_lr,
                                   csv_logger])

def toArray(tensor):
    arr = [[0 for x in range(50)] for y in range(50)]
    for x in range(50):
        for y in range(50):
            arr[x][y] = tensor[0, 0, x, y, 0]
    return arr

if mode_test:
    input = np.ndarray(shape=(10,20,50,50,2))
    for i in range(10):
        input[i, :, :, :, :] = np.load('data/0.npy')  # step i data
    output = model.predict(input)
    out = [[0 for x in range(50)] for y in range(50)]
    print(out)
    exit(0)

tf = time.time()
time_file = np.array([t0, tf, tf - t0])
np.save('models/sensor_review_time' + str(initial_epoch) + '.npy', time_file)
