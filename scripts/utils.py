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

@keras.saving.register_keras_serializable()
class ReflectionPadding2D(tf.keras.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

# Make Gaussian kernel following SciPy logic
def make_gaussian_2d_kernel(sigma, truncate=4.0):
    radius = tf.cast(sigma * truncate, tf.int32)
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    k = tf.exp(-0.5 * tf.square(x / sigma))
    k = k / tf.reduce_sum(k)
    k = tf.expand_dims(k, 1) * k
    return k[..., tf.newaxis, tf.newaxis]


def tf_gaussian_filter(image, sigma):
    kernel = make_gaussian_2d_kernel(sigma)
    image = ReflectionPadding2D(padding=kernel.shape[0] // 2)(image)
    return tf.nn.conv2d(image, kernel, strides=1, padding='VALID')

def ks93(g1, g2):
    # equation 5.62 from Introduction to Gravitational Lensing Lecture scripts (Massimo Meneghetti)

    (nx, ny) = g1.shape

    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    g1hat = np.fft.fft2(g1)
    g2hat = np.fft.fft2(g2)

    khat = ((k1**2-k2**2)*g1hat + 2*k1*k2*g2hat) / (k1**2+k2**2+1e-20)

    kappa_fft = np.fft.ifft2(khat).real
    kappa_fft = sp_gaussian_filter(kappa_fft, 2)
    return kappa_fft

