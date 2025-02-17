# form https://www.tensorflow.org/tutorials/generative/pix2pix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython import display
from tqdm.notebook import tqdm

from scipy.ndimage import gaussian_filter as sp_gaussian_filter

import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import h5py
import json
import time
import random

import keras

from utils import *

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D
         (filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def build_unet(
        kernel_size=2,        # optimized
        min_filters=64,       # default
        max_filters=512,      # default
        extend_borders=True,  # optimized
        extra_layer=True,     # optimized
        padding_edge=64,      # fixed
        make_ks93=False
        ):
    if not extend_borders:
        extra_layer = False

    if make_ks93:
        input_channels = 4
    else:
        input_channels = 3

    num_layers = 7
    if extra_layer:
        num_layers += 1
    num_filters = []
    filters = min_filters
    for i in range(num_layers):
        num_filters.append(min(filters, max_filters))
        filters *= 2

    batch_norm = [False] + [True] * (num_layers - 1)
    dropout = [True] * 3 + [False] * (num_layers - 4)

    inputs = layers.Input(shape=(128, 128, input_channels))

    reflect_padding = ReflectionPadding2D(padding=padding_edge)
    cropping = layers.Cropping2D(cropping=padding_edge)

    down_stack = [
        downsample(num_filters[i],  kernel_size, apply_batchnorm=batch_norm[i])
        for i in range(num_layers)
        ]

    up_stack = [
        upsample(num_filters[-i], kernel_size, apply_dropout=dropout[i-1])
        for i in range(1, num_layers)
        ]

    # creo que lo anterior era incorrecto (respecto a la arquitectura original)
    # deberia ser lo siguiente, pero habria que entrenar de cero
    # up_stack = [
    #     upsample(num_filters[-i-1], kernel_size, apply_dropout=dropout[i-1])
    #     for i in range(1, num_layers-1)
    #     ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, kernel_size,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')
    x = inputs

    if extend_borders:
        x = reflect_padding(x)

    # Downsampling through the model
    skips = []
    for down in down_stack[:-1]:
        x = down(x)
        skips.append(x)
    x = down_stack[-1](x)

    skips = list(reversed(skips))

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    if extend_borders:
        x = cropping(x)

    return Model(inputs=inputs, outputs=x)
