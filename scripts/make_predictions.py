import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython import display
from keras.callbacks import CSVLogger
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
import datetime


from models import *
from utils import *
from loss import *
from data import *
from metrics import *
import logging

import shutil


def save_predictions_zip(k_pred, filenames, outdir, team="barrienlens"):

    tmp_dir = f"/storage/jhernand.2024/dataton/wl_dataton/tmp/{outdir}"
    if not os.path.exists(f"{tmp_dir}/{team}"):
        os.makedirs(f"{tmp_dir}/{team}")
    # clean tmp directory
    for f in os.listdir(f"{tmp_dir}/{team}"):
        shutil.rmtree(os.path.join(f"{tmp_dir}/{team}", f))

    savedir = f"/storage/jhernand.2024/dataton/wl_dataton/predictions/{outdir}"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #else:
    #    raise ValueError(f"Output directory {outdir} already exists")

    print("saving results into temp dir")
    for i, k in enumerate(k_pred):
        k = np.squeeze(k).astype(np.float16)
        np.save(f"{tmp_dir}/{team}/{filenames[i]}", k)

    print("zipping files")
    cwd = os.getcwd()
    os.chdir(f"{tmp_dir}")
    os.system(f'zip -r {team}_submission.zip {team}')
    os.chdir(cwd)

    print("copying to savedir")
    shutil.copy(f"{tmp_dir}/{team}_submission.zip", savedir)

    # print("cleaning tmp dir")
    # shutil.rmtree(f"/content/tmp/{outdir}")

    print("done")

with open("/storage/jhernand.2024/dataton/wl_dataton/h5_files/val/8.json") as f:
    filenames = json.load(f)
    filenames = [os.path.basename(f) for f in filenames]


model_name = "hyperunet_dms_loss"


model = build_unet(
            kernel_size=2,
            extend_borders=True,
            extra_layer=True,
            make_ks93=False
        )
restart_epoch = 100

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    loss=WMSELoss(),
    # loss=CombinedLoss(),
    metrics=[
        tf.keras.metrics.MeanSquaredError(name="mse", dtype=None),
        WMAPEMetric(),
        DICEEMetric(),
        DPEAKSMetric(n_peaks=3),
    ],
)
model.load_weights(f"/storage/jhernand.2024/dataton/wl_dataton/models/{model_name}/{model_name}_{restart_epoch}.keras")


trained_model = model

test_dataset, num_test_batches = create_test_dataset_h5(base_dir='/storage/jhernand.2024/dataton/wl_dataton/h5_files/val',
                                                        batch_size=32, make_ks93=False)
test_k_pred = trained_model.predict(test_dataset, steps=num_test_batches)


import shutil

save_predictions_zip(test_k_pred, filenames, model_name)
