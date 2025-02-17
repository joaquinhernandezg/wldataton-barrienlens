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
from loss import *
from data import *

@keras.saving.register_keras_serializable()
class FocalLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name='focal_loss', eps=1e-5, **kwargs):
        super(FocalLossMetric, self).__init__(name=name, **kwargs)
        self.focal_loss_sum = self.add_weight(name='focal_loss_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.eps = eps  # Small constant to avoid division by zero

    def weights(self, y_true):
        max_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        weights = 1 + y_true / tf.maximum(max_true, self.eps)
        return weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = self.weights(y_true)
        sq_diff = tf.square(y_true - y_pred)
        batch_focal_loss = tf.reduce_sum(sq_diff * weights, axis=(1, 2, 3))
        self.focal_loss_sum.assign_add(tf.reduce_sum(batch_focal_loss))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.focal_loss_sum / self.total_samples

    def reset_state(self):
        self.focal_loss_sum.assign(0.)
        self.total_samples.assign(0.)


@keras.saving.register_keras_serializable()
class MultiFocalLossMetric(tf.keras.metrics.Metric):
    def __init__(self,
                 weight_focal=1.0,
                 weight_local=1.0,
                 weight_peak=1.0,
                 weight_structure=1.0,
                 weight_edges=0.3,
                 box_size=5,
                 thresh_quantile=0.7,
                 num_peaks=5,
                 peak_radius=3,
                 edge_size=5,
                 eps=1e-5,
                 name='multi_focal_loss', **kwargs):
        super(MultiFocalLossMetric, self).__init__(name=name, **kwargs)
        self.multi_focal_loss = MultiFocalLoss(weight_focal, weight_local, weight_peak, weight_structure, weight_edges,
                                               box_size, thresh_quantile, num_peaks, peak_radius, edge_size, eps)
        self.loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_loss = self.multi_focal_loss(y_true, y_pred)
        self.loss_sum.assign_add(tf.reduce_sum(batch_loss))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.loss_sum / self.total_samples

    def reset_state(self):
        self.loss_sum.assign(0.)
        self.total_samples.assign(0.)


@keras.saving.register_keras_serializable()
class WMAPEMetric(tf.keras.metrics.Metric):
    def __init__(self, name='wmape', eps=1e-5, **kwargs):
        super(WMAPEMetric, self).__init__(name=name, **kwargs)
        self.wmape_sum = self.add_weight(name='wmape_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.eps = eps  # Small constant to avoid division by zero

    def weights(self, y_true):
        max_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True)
        weights = 1 + y_true / tf.maximum(max_true, self.eps)
        return weights

    def _score_fn(self, y_true, y_pred):
        y_true = tf.where(y_true == 0, self.eps, y_true)
        y_pred = tf.where(y_pred == 0, self.eps, y_pred)
        weights = self.weights(y_true)
        rel_diff = tf.abs(y_true - y_pred) / tf.abs(y_true)
        return tf.keras.ops.average(rel_diff, axis=(1, 2, 3), weights=weights)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_wmape = self._score_fn(y_true, y_pred)
        self.wmape_sum.assign_add(tf.reduce_sum(batch_wmape))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.wmape_sum / self.total_samples

    def reset_state(self):
        self.wmape_sum.assign(0.)
        self.total_samples.assign(0.)


@keras.saving.register_keras_serializable()
class DICEEMetric(tf.keras.metrics.Metric):
    def __init__(self, alpha=0.5, beta=0.5, name='dicee', **kwargs):
        super(DICEEMetric, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.dicee_sum = self.add_weight(name='dicee_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def _score_fn(self, y_true, y_pred):
        kappa_mean_true = tf.reduce_mean(y_true, axis=(1, 2, 3), keepdims=True)
        kappa_mean_pred = tf.reduce_mean(y_pred, axis=(1, 2, 3), keepdims=True)
        structure_true = tf.where(y_true > kappa_mean_true, 1.0, 0.0)
        structure_pred = tf.where(y_pred > kappa_mean_pred, 1.0, 0.0)
        intersects = tf.reduce_sum(structure_true * structure_pred, axis=(1, 2, 3))
        diff_true = tf.reduce_sum((1 - structure_true) * structure_pred, axis=(1, 2, 3))
        diff_pred = tf.reduce_sum((1 - structure_pred) * structure_true, axis=(1, 2, 3))
        union = intersects + self.alpha * diff_true + self.beta * diff_pred
        return 1 - intersects / union

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_dicee = self._score_fn(y_true, y_pred)
        self.dicee_sum.assign_add(tf.reduce_sum(batch_dicee))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.dicee_sum / self.total_samples

    def reset_state(self):
        self.dicee_sum.assign(0.)
        self.total_samples.assign(0.)


@keras.saving.register_keras_serializable()
class DPEAKSMetric(tf.keras.metrics.Metric):
    def __init__(self, n_peaks=3, normalize=False, name='dpeaks', **kwargs):
        super(DPEAKSMetric, self).__init__(name=name, **kwargs)
        self.n_peaks = n_peaks
        self.dpeaks_sum = self.add_weight(name='dpeaks_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.normalize = normalize

    def _score_fn(self, y_true, y_pred):
        _, height, width, _ = y_true.shape
        true_x, true_y = find_n_max_clump_positions(y_true, self.n_peaks)
        pred_x, pred_y = find_n_max_clump_positions(y_pred, self.n_peaks)
        loss_x = tf.abs(true_x - pred_x)
        loss_y = tf.abs(true_y - pred_y)
        dist = tf.reduce_sum(loss_x + loss_y, axis=1)
        dpeaks = tf.cast(dist, tf.float32)
        if self.normalize:
            dpeaks = dpeaks / (self.n_peaks * (height + width))
        return dpeaks

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_dpeaks = self._score_fn(y_true, y_pred)
        self.dpeaks_sum.assign_add(tf.reduce_sum(batch_dpeaks))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.dpeaks_sum / self.total_samples

    def reset_state(self):
        self.dpeaks_sum.assign(0.)
        self.total_samples.assign(0.)

@keras.saving.register_keras_serializable()
class CombinedMetric(tf.keras.metrics.Metric):
    def __init__(self, metrics, weights=None, name='combined_metric', **kwargs):
        super(CombinedMetric, self).__init__(name=name, **kwargs)
        self.metrics = metrics
        if weights is None:
            self.weights = tf.ones(len(metrics), dtype=tf.float32)
        else:
            self.weights = tf.cast(weights, tf.float32)
        self.weights = self.weights / len(self.metrics)
        self.combined_sum = self.add_weight(name='combined_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        combined_value = 0
        for i, m in enumerate(self.metrics):
            # m.update_state(y_true, y_pred, sample_weight)  # Update individual metrics
            # combined_value += m.result() * self.weights[i] * m.total_samples
            combined_value += m._score_fn(y_true, y_pred) * self.weights[i]
        self.combined_sum.assign_add(tf.reduce_sum(combined_value))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.combined_sum / self.total_samples

    def reset_state(self):
        self.combined_sum.assign(0.)
        self.total_samples.assign(0.)
        for m in self.metrics:
            m.reset_state()  # Reset individual metrics

    def get_config(self):
        config = super(CombinedMetric, self).get_config()
        # Serialize the list of metrics by their configuration
        config.update({
            'metrics': [tf.keras.utils.serialize_keras_object(m) for m in self.metrics],
            'weights': self.weights.numpy().tolist(),  # Serialize the weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the list of metrics from their configurations
        metrics = config.pop('metrics')
        metrics = [tf.keras.utils.deserialize_keras_object(m) for m in metrics]
        weights = config.pop('weights', None)
        return cls(metrics=metrics, weights=weights, **config)

@keras.saving.register_keras_serializable()
def ModelScore(target_wmape=2.0, target_dice=0.06, target_dpeaks=30):
    return CombinedMetric([WMAPEMetric(), DICEEMetric(), DPEAKSMetric(n_peaks=3, normalize=False)],
                          weights=[1/target_wmape, 1/target_dice, 1/target_dpeaks],
                          name='model_score')