import math
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K


def leaky_relu(x):
    return K.relu(x, alpha=0)


def whitten(x):
    temp = keras.layers.concatenate([K.ones_like(x[:, :10, :, :, :]), K.zeros_like(x[:, 10:, :, :, :])], axis=1)
    return x * temp


def whitten_multi(x):
    t1 = K.ones_like(x[:, :, :10, :, :, :])
    t2 = K.ones_like(x[:, :, 10:, :, :, :])
    temp = keras.layers.concatenate([K.ones_like(x[:, :, :10, :, :, :]), K.zeros_like(x[:, :, 10:, :, :, :])], axis=2)
    return x * temp

def crop_input(input_layer, n_labels):
    temp = keras.layers.Reshape((20, 64, 64, n_labels + 1))(input_layer[:, :, :, :, :(n_labels + 1)])
    return temp


def loss_fun(mask, labels):
    # mask = keras.layers.concatenate([mask for i in range(labels)], axis = 4)
    def loss(y_true, y_pred):
        new_pred = keras.layers.multiply([mask, y_pred])
        return K.binary_crossentropy(y_true, new_pred)

    return loss


def loss_fun_flat(mask, width, height):
    mask = keras.layers.Reshape((10, width, height, 1))(mask)

    def loss(y_true, y_pred):
        new_pred = keras.layers.multiply([mask, y_pred])
        return K.binary_crossentropy(y_true, new_pred)

    return loss


def loss_fun_flat_collab(mask, out2, lam=0.01):
    mask = keras.layers.Reshape((10, 64, 64, 1))(mask)
    regularizer = lam * (K.sum(out2) - 1300)

    def loss(y_true, y_pred):
        new_pred = keras.layers.multiply([mask, y_pred])
        new_true = keras.layers.multiply([mask, y_true])
        return K.binary_crossentropy(new_true, new_pred) + regularizer

    return loss


# OPS
def filter_input(x, input_layer_remote):
    x = keras.layers.concatenate([x for i in range(2)], axis=4)
    new_input = keras.layers.Multiply()([input_layer_remote, x])
    return new_input


def range_inputs(x):
    x = -x + 0.5
    x = keras.layers.Activation(activation='relu')(x)
    x = -2 * x + 1
    x = keras.layers.Activation(activation='relu')(x)
    return (x)
