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
from models import *
from utils import *

def get_random_example(hdf5_files, make_ks93=False):
    file_path = random.choice(hdf5_files)
    with h5py.File(file_path, 'r') as f:
        idx = random.randint(0, len(f['epsilon']))
        epsilon = f['epsilon'][idx].astype(np.float32)
        kappa = f['kappa'][idx].astype(np.float32)
        if make_ks93:
            e1, e2 = epsilon[..., 0], epsilon[..., 1]
            kappa_ks93 = ks93(e1, e2)
            kappa_ks93 = np.expand_dims(kappa_ks93, axis=-1).astype(np.float32)
            epsilon = np.concatenate([epsilon, kappa_ks93], axis=-1)
    return epsilon, kappa, idx


def get_example(hdf5_files, idx, make_ks93=False):
    file_path = random.choice(hdf5_files)
    with h5py.File(file_path, 'r') as f:
        epsilon = f['epsilon'][idx].astype(np.float32)
        kappa = f['kappa'][idx].astype(np.float32)
        if make_ks93:
            e1, e2 = epsilon[..., 0], epsilon[..., 1]
            kappa_ks93 = ks93(e1, e2)
            kappa_ks93 = np.expand_dims(kappa_ks93, axis=-1).astype(np.float32)
            epsilon = np.concatenate([epsilon, kappa_ks93], axis=-1)
    return epsilon, kappa


def load_hdf5_train(file_path, chunk_size):
    with h5py.File(file_path, 'r') as f:
        for i in range(0, len(f['epsilon']), chunk_size):
            epsilon = f['epsilon'][i:i+chunk_size]
            kappa = f['kappa'][i:i+chunk_size]
            yield epsilon, kappa


def data_generator_train(hdf5_files, chunk_size=1000, make_ks93=False):
    for file_path in hdf5_files:
        for epsilon, kappa in load_hdf5_train(file_path, chunk_size):
            for i in range(len(epsilon)):
                if make_ks93:
                    e1, e2 = epsilon[i, :, :, 0], epsilon[i, :, :, 1]
                    kappa_ks93 = ks93(e1, e2)
                    kappa_ks93 = np.expand_dims(kappa_ks93, axis=-1)
                    augmented_epsilon = np.concatenate([epsilon[i], kappa_ks93], axis=-1)
                    # smoothing data
                    augmented_epsilon = sp_gaussian_filter(augmented_epsilon, 2)
                    yield augmented_epsilon, kappa[i]

                else:
                    yield epsilon[i], kappa[i]


def load_hdf5_test(file_path, chunk_size):
    with h5py.File(file_path, 'r') as f:
        for i in range(0, len(f['epsilon']), chunk_size):
            epsilon = f['epsilon'][i:i+chunk_size]
            yield epsilon


def data_generator_test(hdf5_files, chunk_size=1000, make_ks93=False):
    for file_path in hdf5_files:
        for epsilon in load_hdf5_test(file_path, chunk_size):
            for i in range(len(epsilon)):
                if make_ks93:
                    e1, e2 = epsilon[i, :, :, 0], epsilon[i, :, :, 1]
                    kappa_ks93 = ks93(e1, e2)
                    kappa_ks93 = np.expand_dims(kappa_ks93, axis=-1)
                    augmented_epsilon = np.concatenate([epsilon[i], kappa_ks93], axis=-1)
                    augmented_epsilon = tf_gaussian_filter(augmented_epsilon, 2)
                    yield augmented_epsilon
                else:
                    yield epsilon[i]


def make_augment_function(
        ks93_remove_prob=0.5,
        ):

    @tf.function
    def augment_sample(epsilon, kappa):
        # Generate random rotation and flip flags
        rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # Random rotations: 0, 90, 180, 270 degrees

        # # Apply the same rotation to both epsilon and kappa
        epsilon = tf.image.rot90(epsilon, k=rotations)
        kappa = tf.image.rot90(kappa, k=rotations)

        # Randomly flip both epsilon and kappa in the same way
        flip_left_right = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        flip_up_down = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)

        if flip_left_right == 1:
            epsilon = tf.image.flip_left_right(epsilon)
            kappa = tf.image.flip_left_right(kappa)

        if flip_up_down == 1:
            epsilon = tf.image.flip_up_down(epsilon)
            kappa = tf.image.flip_up_down(kappa)

        if tf.shape(epsilon)[-1] == 4: # has ks93
            epsilon, kappa_ks93 = epsilon[..., :-1], epsilon[..., -1:]
            # random remove ks93 so that the model does not rely on it
            if tf.random.uniform(shape=[]) < ks93_remove_prob:
                kappa_ks93 = tf.zeros_like(kappa_ks93)

            epsilon = tf.concat([epsilon, kappa_ks93], axis=-1)


        return epsilon, kappa

    return augment_sample


def create_dataset_h5(
        hdf5_files,
        batch_size=32,
        data_fraction=1.0,
        shuffle=True,
        augment=True,
        with_labels=True,
        make_ks93=False,
        ks93_remove_prob=0.5,
        chunk_size=1000
        ):

    if shuffle:
        random.shuffle(hdf5_files)

    num_elements = 0
    for file_path in hdf5_files:
        with h5py.File(file_path, 'r') as f:
            num_elements += len(f['epsilon'])
    num_elements = int(num_elements * data_fraction)

    num_batches = int(np.ceil(num_elements / batch_size))

    if make_ks93:
        input_channels = 4
    else:
        input_channels = 3

    # Create a / from the generator
    if with_labels:
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator_train(hdf5_files, chunk_size, make_ks93),
            output_signature=(
                tf.TensorSpec(shape=(128, 128, input_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32)
            ),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator_test(hdf5_files, chunk_size, make_ks93),
            output_signature=(
                tf.TensorSpec(shape=(128, 128, input_channels), dtype=tf.float32)
            ),
        )

    dataset = dataset.take(num_elements)

    if augment:
        # Apply the transformations
        dataset = dataset.map(make_augment_function(ks93_remove_prob),
                              num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch the data for efficiency
    if shuffle:
        dataset = dataset.shuffle(buffer_size=chunk_size)
    if with_labels:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, num_batches


def create_train_dataset_h5(base_dir, batch_size=32, data_fraction=1.0,
                            make_ks93=False, ks93_remove_prob=0.5,
                            chunk_size=1000):
    hdf5_files = glob.glob(os.path.join(base_dir, '*.h5'), recursive=True)

    dataset, num_batches = create_dataset_h5(
        hdf5_files,
        batch_size=batch_size,
        make_ks93=make_ks93,
        ks93_remove_prob=ks93_remove_prob,
        chunk_size=chunk_size,
        data_fraction=data_fraction,
        shuffle=True,
        augment=True,
        with_labels=True)

    return dataset, num_batches


def create_val_dataset_h5(base_dir, batch_size=32, data_fraction=1.0,
                          make_ks93=False, chunk_size=1000):
    hdf5_files = glob.glob(os.path.join(base_dir, '*.h5'), recursive=True)

    dataset, num_batches = create_dataset_h5(
        hdf5_files,
        batch_size=batch_size,
        make_ks93=make_ks93,
        chunk_size=chunk_size,
        data_fraction=data_fraction,
        shuffle=False,
        augment=False,
        with_labels=True)

    return dataset, num_batches


def create_test_dataset_h5(base_dir, batch_size=32, make_ks93=False, chunk_size=1000):
    hdf5_files = glob.glob(os.path.join(base_dir, '*.h5'), recursive=True)
    hdf5_files = sorted(hdf5_files)

    dataset, num_batches = create_dataset_h5(
        hdf5_files,
        batch_size=batch_size,
        make_ks93=make_ks93,
        chunk_size=chunk_size,
        data_fraction=1.0,
        shuffle=False,
        augment=False,
        with_labels=False)

    return dataset, num_batches