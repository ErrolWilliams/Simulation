import pandas as pd
import numpy as np

#############  KERAS OBJECTS
# from __future__ import print_function
import keras

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Conv3D, TimeDistributed, ConvLSTM2D, Cropping3D
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import CSVLogger, ReduceLROnPlateau

from ops import *


def model_labels(input_shape, learning_rate, metrics=['mse'], training=True):
    # Input layer
    n_labels = int((input_shape[3] - 1) / 2)
    # input comes: 4 digits occ -- mask -- 4 dig vis

    input_layer = Input(shape=input_shape, name='input_layer')
    # loss mask: 0 layer from input (visibility frame)
    mask = keras.layers.Reshape((10, 64, 64, n_labels))(input_layer[:, 10:, :, :, (n_labels + 1):(2 * n_labels) + 1])

    real_input = keras.layers.Lambda(crop_input, arguments={'n_labels': n_labels})(input_layer)
    # whitten 10 last frames for training
    if training:
        x = keras.layers.Lambda(whitten)(real_input)
    else:
        x = real_input
    # Embedding
    x = TimeDistributed(Conv2D(8, kernel_size=(7, 7),
                               activation=leaky_relu,
                               padding='same'))(x)
    # Belief tracker
    x = ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                   activation=leaky_relu,
                   return_sequences=True)(x)
    # keep only last 10 frames (predictions)
    x = Cropping3D(((10, 0), (0, 0), (0, 0)))(x)
    # Reconstruct output
    x = TimeDistributed(Conv2D(n_labels, (7, 7),
                               padding='same'))(x)
    # Sigmoid for logistic output
    out = keras.layers.Activation(activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    if not training:
        for layer in model.layers:
            layer.trainable = False
    model.compile(loss=loss_fun(mask, n_labels),
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=metrics)
    return model


def model_flat(input_shape, learning_rate, training=True):
    input_layer = Input(shape=input_shape, name='input_layer')
    # For the flat model occupancy is index 0
    if training:
        x = keras.layers.Lambda(whitten)(input_layer)
    else:
        x = input_layer
    mask = keras.layers.Reshape((10, 50, 50, 1))(input_layer[:, 10:, :, :, 1])  # last 10 visibility grids
    x = keras.layers.Lambda(whitten)(input_layer)
    x = TimeDistributed(Conv2D(8, kernel_size=(7, 7),
                               activation=leaky_relu,
                               padding='same'))(x)
    x = ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                   activation=leaky_relu,
                   return_sequences=True)(x)
    x = Cropping3D(((10, 0), (0, 0), (0, 0)))(x)
    x = TimeDistributed(Conv2D(1, (7, 7),
                               padding='same'))(x)
    out = keras.layers.Activation(activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    if not training:
        for layer in model.layers:
            layer.trainable = False
    model.compile(loss=loss_fun_flat(mask),
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['mse'])
    return model


"""
def model_flat_multi(input_shape, learning_rate, training=True):
    input_layer = Input(shape=input_shape, name='input_layer')
    # For the flat model occupancy is index 0
    if training:
        x = keras.layers.Lambda(whitten_multi)(input_layer)
    else:
        x = input_layer
    mask = keras.layers.Reshape((input_shape[0], 10, 50, 50, 1))(input_layer[:, :, 10:, :, :, 1])  # last 10 visibility grids
    x = keras.layers.Lambda(whitten_multi)(input_layer)
    x = TimeDistributed(Conv3D(8, kernel_size=(7, 7, 7),
                               activation=leaky_relu,
                               padding='same',
                               data_format='channels_last'))(x)
    x = ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                   activation=leaky_relu,
                   return_sequences=True,
                   data_format='channels_last')(x)
    x = Cropping3D(((10, 0), (0, 0), (0, 0)))(x)
    x = TimeDistributed(Conv3D(1, (7, 7, 7),
                               padding='same'))(x)
    out = keras.layers.Activation(activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    if not training:
        for layer in model.layers:
            layer.trainable = False
    model.compile(loss=loss_fun_flat(mask),
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['mse'])
    return model
    """


def model_col(input_shape, learning_rate):
    input_layer_remote = Input(shape=input_shape, name='remote_input')
    x = TimeDistributed(Conv2D(8, kernel_size=(7, 7),
                               activation='relu',
                               padding='same'), name='remote_1')(input_layer_remote)
    x = TimeDistributed(Conv2D(16, kernel_size=(7, 7),
                               activation='relu',
                               padding='same'), name='remote_2')(x)
    x = TimeDistributed(Conv2D(1, kernel_size=(7, 7),
                               activation='sigmoid',
                               padding='same'), name='remote_3')(x)
    # x = keras.layers.Activation(activation = 'tanh', name = 'enforce_range')(x)
    # x = keras.layers.Activation(activation = 'relu', name = 'positive_filters')(x)
    out_remote = keras.layers.Lambda(filter_input, arguments={'input_layer_remote': input_layer_remote},
                                     name='filter_input_remote')(x)

    input_layer_local = Input(shape=input_shape, name='input_layer_local')
    new_input = keras.layers.Add(name='concat_inputs')([input_layer_local, out_remote])
    new_input = keras.layers.Lambda(range_inputs, name='normalize_inputs')(new_input)

    mask = keras.layers.Reshape((10, 64, 64, 1))(new_input[:, 10:, :, :, 1])

    x2 = TimeDistributed(Conv2D(8, kernel_size=(7, 7),
                                activation=leaky_relu,
                                padding='same'), trainable=False, name='local_1')(new_input)
    x2 = ConvLSTM2D(16, kernel_size=(5, 5), padding='same',
                    activation=leaky_relu,
                    return_sequences=True, trainable=False, name='local_2')(x2)
    x2 = Cropping3D(((10, 0), (0, 0), (0, 0)))(x2)
    x2 = TimeDistributed(Conv2D(1, (7, 7),
                                padding='same'), trainable=False, name='local_3')(x2)
    out = keras.layers.Activation(activation='sigmoid')(x2)

    model = Model(inputs=[input_layer_local, input_layer_remote], outputs=[out])

    model.compile(loss=loss_fun_flat_collab(mask),
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['mse'])
    return model
