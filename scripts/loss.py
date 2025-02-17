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
from models import *
from data import *

def find_n_max_pixel_positions(image, n):
    _, height, width, _ = image.shape
    flattened_image = tf.reshape(image, (-1, height * width))
    _, indices = tf.nn.top_k(flattened_image, k=n)
    y_coords = indices // width
    x_coords = indices % width
    return x_coords, y_coords


def mask_edge(image, n_edge):
    shape = tf.shape(image)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    mask = tf.zeros_like(image)
    edge_mask = tf.ones_like(image)
    edge_mask_top_bottom = tf.concat([tf.ones((batch_size, n_edge, width, channels)),
                                    tf.zeros((batch_size, height - 2 * n_edge, width, channels)),
                                    tf.ones((batch_size, n_edge, width, channels))], axis=1)
    edge_mask_left_right = tf.concat([tf.ones((batch_size, height, n_edge, channels)),
                                    tf.zeros((batch_size, height, width - 2 * n_edge, channels)),
                                    tf.ones((batch_size, height, n_edge, channels))], axis=2)
    mask = tf.maximum(edge_mask_top_bottom, edge_mask_left_right)
    return mask


def find_n_max_clump_positions(image, n, box_size=3, smooth=3, edge=1):
    smoothed = tf_gaussian_filter(image, smooth)
    clumps = tf.nn.max_pool(smoothed, box_size, 1, padding='SAME')
    masked_image = tf.where(smoothed == clumps, smoothed, tf.zeros_like(image))
    edge_mask = mask_edge(masked_image, edge)
    masked_image = tf.where(edge_mask == 0, masked_image, tf.zeros_like(image))
    x_coords, y_coords = find_n_max_pixel_positions(masked_image, n)
    return x_coords, y_coords

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


def find_top_k_peaks_oficial(im ,sigma=3, N=3):
    smoothed = gaussian_filter(im, sigma=sigma)
    coordinates = peak_local_max(smoothed, threshold_abs=None, num_peaks=N)
    while len(coordinates) < 3: coordinates = np.vstack([coordinates, np.array([0,0])])
    return coordinates.T[1], coordinates.T[0]

def DPEAKS_oficial(T:np.ndarray, P:np.ndarray, num_peaks = 3):
    PEAKS_T = find_top_k_peaks_oficial(T, N=num_peaks)
    PEAKS_P = find_top_k_peaks_oficial(P, N=num_peaks)
    sum_DPEAKS = np.sum(np.abs(PEAKS_T-PEAKS_P))
    return sum_DPEAKS

def create_W(T):
    abs_min_T = tf.abs(tf.reduce_min(T, axis=[2, 3], keepdims=True))
    abs_max_T = tf.abs(tf.reduce_max(T, axis=[2, 3], keepdims=True))
    t_max_t = (T + abs_min_T) / (abs_max_T + abs_min_T)
    W = 1.0 + t_max_t
    return W

# WMSE: Function that calculates the weighted mean squared error for a batch of predicted data (P)
# with respect to its ground truth batch (T).
# T: Tensor "Ground Truth" of shape (b, 1, 128, 128), where b is the batch size. Contains b convergence maps.
# P: Tensor "Predictions" of shape (b, 1, 128, 128), where b is the batch size. Contains b convergence maps.
# Returns: Tensor "WMSE" of shape (b, 1, 1, 1) with the weighted mean squared error result for each of the b convergence maps.
def WMSE(T, P):
    W = create_W(T)
    Numerator = W * tf.math.square(P - T)
    sum_Numerator = tf.reduce_sum(Numerator, axis=[2, 3], keepdims=True)
    sum_W = tf.reduce_sum(W, axis=[2, 3], keepdims=True)
    return sum_Numerator / sum_W

@keras.saving.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, eps=1e-7, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.eps = eps  # Small constant to avoid division by zero

    def weights(self, y_true):
        # weight is proportional to absolute peak
        max_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        weights = y_true / tf.maximum(max_true, self.eps)
        return (weights + 1.0) / 2

    def call(self, y_true, y_pred):
        weights = self.weights(y_true)
        sq_diff = tf.square(y_true - y_pred)
        return tf.reduce_sum(sq_diff * weights, axis=(1, 2, 3))





# WMSELoss: Class for the WMSE loss function that returns the sum of the weighted mean squared error in a batch.
@keras.saving.register_keras_serializable()
class WMSELoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(WMSE(y_true, y_pred))

@keras.saving.register_keras_serializable()
class MultiFocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 weight_focal=1.0,
                 weight_local=2.0,
                 weight_peak=3.0,
                 weight_structure=1.5,
                 weight_edges=1.0,
                 local_box_size=8,
                 peaks_box_size=3,
                 thresh_quantile=0.7,
                 structure_width=0.01,
                 num_peaks=5,
                 peak_radius=3,
                 edge_size=5,
                 eps=1e-5,
                 **kwargs):
        super(MultiFocalLoss, self).__init__(**kwargs)
        weight_sum = weight_focal + weight_local + weight_peak + weight_structure + weight_edges
        self.weight_focal = weight_focal / weight_sum
        self.weight_local = weight_local / weight_sum
        self.weight_peak = weight_peak / weight_sum
        self.weight_structure = weight_structure / weight_sum
        self.weight_edges = weight_edges / weight_sum
        self.local_box_size = local_box_size
        self.peaks_box_size = peaks_box_size
        self.quantile = thresh_quantile
        self.structure_width = structure_width
        self.num_peaks = num_peaks
        self.peak_radius = peak_radius
        self.edge_size = edge_size
        self.eps = eps  # Small constant to avoid division by zero

    def weights_focal(self, y_true):
        # weight is proportional to absolute peak
        max_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        weights = y_true / tf.maximum(max_true, self.eps)
        return (weights + 1.0) / 2

    def weights_local(self, y_true):
        # weight is relative to local peak over quantile
        thresh = tf.keras.ops.quantile(y_true, self.quantile,
                                       axis=(1, 2, 3), keepdims=True)
        thresh = tf.maximum(thresh, self.eps)
        clumps = tf.nn.max_pool(y_true, self.local_box_size, 1, padding='SAME')
        weights = (y_true / tf.math.maximum(clumps, self.eps))
        weights = tf.where(y_true > thresh, weights, 0.0)
        return (weights + 1.0) / 2

    def weights_peak(self, y_true):
        # draw circles around peaks
        peak_x, peak_y = find_n_max_clump_positions(y_true, self.num_peaks, self.peaks_box_size)  # batch, num_peaks
        peak_x = tf.expand_dims(tf.expand_dims(peak_x, axis=1), 1)  # batch, w, h, num_peaks
        peak_y = tf.expand_dims(tf.expand_dims(peak_y, axis=1), 1)
        grid_x, grid_y = tf.meshgrid(tf.range(y_true.shape[2]), tf.range(y_true.shape[1]))
        grid_x = tf.expand_dims(tf.expand_dims(grid_x, axis=0), -1)  # batch, w, h, num_peaks
        grid_y = tf.expand_dims(tf.expand_dims(grid_y, axis=0), -1)
        weights = tf.zeros(grid_x.shape)
        distance = tf.abs(grid_x - peak_x) + tf.abs(grid_y - peak_y)
        weights = tf.where(distance < self.peak_radius, 1.0, weights)
        weights = tf.reduce_sum(weights, axis=-1, keepdims=True)
        return tf.minimum(weights, 1.0)

    def weights_structure(self, y_true):
        # weights the edges of the structures near the mean
        true_mean = tf.reduce_mean(y_true, axis=(1, 2, 3), keepdims=True)
        # structure_width = self.structure_width * tf.abs(true_mean)
        strcuture_min = true_mean - self.structure_width
        strcuture_max = true_mean + self.structure_width
        weights = tf.where(y_true > strcuture_min, 1.0, 0.0)
        weights = tf.where(y_true < strcuture_max, weights, 0.0)
        return weights

    def weights_edges(self, y_true):
        # increase weight of image edges
        weights = mask_edge(y_true, self.edge_size)
        return weights

    def weights(self, y_true):
        weights_focal = self.weights_focal(y_true) * self.weight_focal
        weights_local = self.weights_local(y_true) * self.weight_local
        weights_peak = self.weights_peak(y_true) * self.weight_peak
        weights_structure = self.weights_structure(y_true) * self.weight_structure
        weights_edges = self.weights_edges(y_true) * self.weight_edges
        return weights_focal + weights_local + weights_peak + weights_structure + weights_edges

    def call(self, y_true, y_pred):
        weights = self.weights(y_true)
        sq_diff = tf.square(y_true - y_pred)
        return tf.reduce_sum(sq_diff * weights, axis=(1, 2, 3))


@keras.saving.register_keras_serializable()
class WMAPELoss(tf.keras.losses.Loss):

    eps = 1e-5

    def call(self, y_true, y_pred):
        y_true = tf.where(y_true == 0, self.eps, y_true)
        y_pred = tf.where(y_pred == 0, self.eps, y_pred)
        weights = self.weights(y_true)
        rel_diff = tf.abs(y_true - y_pred) / tf.abs(y_true)
        return tf.keras.ops.average(rel_diff, axis=(1, 2, 3), weights=weights)

    def weights(self, y_true):
        max_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        weights = 1 + y_true / tf.maximum(max_true, self.eps)
        return weights


@keras.saving.register_keras_serializable()
class DICEELoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        super(DICEELoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        kappa_mean_true = tf.reduce_mean(y_true, axis=(1, 2, 3), keepdims=True)
        kappa_mean_pred = tf.reduce_mean(y_pred, axis=(1, 2, 3), keepdims=True)
        structure_true = tf.where(y_true > kappa_mean_true, 1.0, 0.0)
        structure_pred = tf.where(y_pred > kappa_mean_pred, 1.0, 0.0)
        intersects = tf.reduce_sum(structure_true * structure_pred, axis=(1, 2, 3))
        diff_true = tf.reduce_sum((1 - structure_true) * structure_pred, axis=(1, 2, 3))
        diff_pred = tf.reduce_sum((1 - structure_pred) * structure_true, axis=(1, 2, 3))
        union = intersects + self.alpha * diff_true + self.beta * diff_pred
        return 1 - intersects / union


@keras.saving.register_keras_serializable()
class DPEAKSLoss(tf.keras.losses.Loss):
    def __init__(self, n_peaks=3, normalize=True, **kwargs):
        super(DPEAKSLoss, self).__init__(**kwargs)
        self.n_peaks = n_peaks
        self.normalize = normalize

    def call(self, y_true, y_pred):
        _, height, width, _ = y_true.shape
        true_x, true_y = find_n_max_clump_positions(y_true, self.n_peaks)
        pred_x, pred_y = find_n_max_clump_positions(y_pred, self.n_peaks)
        loss_x = tf.abs(true_x - pred_x)
        loss_y = tf.abs(true_y - pred_y)
        dist = tf.reduce_sum(loss_x + loss_y, axis=1)
        dist = tf.cast(dist, tf.float32)
        if self.normalize:
            dist = dist / (self.n_peaks * (height + width))
        return dist

@keras.saving.register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, losses, weights=None, **kwargs):
        super(CombinedLoss, self).__init__(**kwargs)
        self.losses = losses
        if weights is None:
            self.weights = tf.ones(len(losses), dtype=tf.float32)
        else:
            self.weights = tf.cast(weights, tf.float32)
        self.weights = self.weights / tf.reduce_sum(self.weights)

    def call(self, y_true, y_pred):
        loss = 0
        for i, l in enumerate(self.losses):
            loss += l(y_true, y_pred) * self.weights[i]
        return loss

    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        # Serialize the list of metrics by their configuration
        config.update({
            'losses': [tf.keras.utils.serialize_keras_object(m) for m in self.losses],
            'weights': self.weights.numpy().tolist(),  # Serialize the weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the list of metrics from their configurations
        losses = config.pop('losses')
        losses = [tf.keras.utils.deserialize_keras_object(m) for m in losses]
        weights = config.pop('weights', None)
        return cls(losses=losses, weights=weights, **config)

def MAE(T, P):
    if P.shape.rank == 3:  # Check the rank of the tensor
        P = tf.reshape(P, (tf.shape(P)[0], 1, tf.shape(P)[1], tf.shape(P)[2]))
    W = tf.ones_like(T)
    Numerator = W * tf.abs(P - T)

    # Ensure we squeeze the correct axis
    if Numerator.shape.rank == 4 and Numerator.shape[1] == 1:  # Check if the second dimension is 1
        Numerator = tf.squeeze(Numerator, axis=1)
        W = tf.squeeze(W, axis=1)

    sum_Numerator = tf.reduce_sum(tf.reshape(Numerator, [tf.shape(Numerator)[0], -1]), axis=1)
    sum_W = tf.reduce_sum(tf.reshape(W, [tf.shape(W)[0], -1]), axis=1)

    return sum_Numerator / sum_W

class MAELoss(tf.keras.losses.Loss):
    def call(self, T, P):
        return tf.reduce_sum(MAE(T, P))

class DMSWMSELoss(tf.keras.losses.Loss):
    def call(self, T, P):
        # Calculate the weight for each pixel
        max_truth = tf.reduce_max(tf.reshape(T, [tf.shape(T)[0], -1]), axis=1, keepdims=True)
        max_truth = tf.reshape(max_truth, [-1, 1, 1, 1])
        weight = 1. + tf.abs(T) / max_truth
        # Calculate the weighted MSE
        wmse = weight * tf.square(P - T)
        loss = tf.reduce_sum(wmse, axis=(1, 2, 3))
        return tf.reduce_sum(loss)

class DMSLoss(tf.keras.losses.Loss):
    def __init__(self, eta_1=1.0, eta_2=1e-3):
        super(DMSLoss, self).__init__()
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.mae_loss = MAELoss()
        self.dmswmse_loss = DMSWMSELoss()

    def call(self, T, P):
        mae_val = self.eta_1 * self.mae_loss(T, P)
        dmswmse_val = self.eta_2 * self.dmswmse_loss(T, P)
        return mae_val + dmswmse_val