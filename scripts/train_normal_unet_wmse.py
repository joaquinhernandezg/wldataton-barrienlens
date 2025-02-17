import tensorflow as tf
from keras.callbacks import CSVLogger

from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter as sp_gaussian_filter

import matplotlib.pyplot as plt
import os
import datetime
import logging

from models import *
from utils import *
from loss import *
from data import *
from metrics import *


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class DynamicLossWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, patience=3, factor=1.2, min_weight=0.1, verbose=1):
        """
        Callback para ajustar dinámicamente los pesos de MultiFocalLoss para minimizar DICEE y WMAPE.

        Parámetros:
        - patience: Número de épocas sin mejora antes de ajustar los pesos.
        - factor: Factor para aumentar los pesos (si empeoran las métricas).
        - min_weight: Valor mínimo permitido para los pesos.
        - verbose: Nivel de mensajes a imprimir (0 = sin mensajes, 1 = mensajes detallados).
        """
        super(DynamicLossWeightScheduler, self).__init__()
        self.patience = patience
        self.factor = factor
        self.min_weight = min_weight
        self.verbose = verbose
        self.best_dicee = float('inf')  # Inicializar con un valor alto (a minimizar)
        self.best_wmape = float('inf')  # Inicializar con un valor alto (a minimizar)
        self.wait_dicee = 0             # Contador de paciencia para DICEE
        self.wait_wmape = 0             # Contador de paciencia para WMAPE

    def on_epoch_end(self, epoch, logs=None):
        # Obtener las métricas actuales
        current_dicee = logs.get('val_dicee')
        current_wmape = logs.get('val_wmape')
        current_dpeaks = logs.get('val_dpeaks')

        # Ajustar pesos si DICEE no mejora (deseamos minimizar DICEE)
        if current_dicee is not None and current_dicee >= self.best_dicee:
            self.wait_dicee += 1

            if self.wait_dicee >= self.patience:
                # Acceder a la instancia de MultiFocalLoss y ajustar los pesos
                loss_instance = self.model.loss
                if isinstance(loss_instance, MultiFocalLoss):
                    # Incrementar peso en Focal y Structure para minimizar DICEE
                    loss_instance.weight_focal = min(loss_instance.weight_focal * self.factor, 1.0)
                    loss_instance.weight_structure = min(loss_instance.weight_structure * self.factor, 1.0)

                    # Reducir peso de peaks solo si DPEAKS es bueno (< 20)
                    if current_dpeaks and current_dpeaks < 20:
                        loss_instance.weight_peak = max(loss_instance.weight_peak / self.factor, self.min_weight)

                    if self.verbose > 0:
                        print(f"\nIncremento de pesos para mejorar DICEE: epoch {epoch+1}")
                        print(f"Nuevo weight_focal: {loss_instance.weight_focal:.4f}")
                        print(f"Nuevo weight_structure: {loss_instance.weight_structure:.4f}")
                        print(f"Nuevo weight_peak: {loss_instance.weight_peak:.4f}")

                # Reiniciar paciencia para DICEE
                self.wait_dicee = 0
        else:
            # Si mejora DICEE, actualizar mejor valor y reiniciar paciencia
            self.best_dicee = current_dicee
            self.wait_dicee = 0

        # Ajustar pesos si WMAPE no mejora (deseamos minimizar WMAPE)
        if current_wmape is not None and current_wmape >= self.best_wmape:
            self.wait_wmape += 1

            if self.wait_wmape >= self.patience:
                # Ajustar pesos de MultiFocalLoss para mejorar WMAPE
                loss_instance = self.model.loss
                if isinstance(loss_instance, MultiFocalLoss):
                    # Incrementar peso en `weight_edges` para ajustar la precisión en regiones con alta masa
                    loss_instance.weight_edges = min(loss_instance.weight_edges * self.factor, 1.0)

                    # Reducir peso de peaks si WMAPE no mejora
                    if current_dpeaks and current_dpeaks < 20:
                        loss_instance.weight_peak = max(loss_instance.weight_peak / self.factor, self.min_weight)

                    if self.verbose > 0:
                        print(f"\nIncremento de pesos para mejorar WMAPE: epoch {epoch+1}")
                        print(f"Nuevo weight_edges: {loss_instance.weight_edges:.4f}")
                        print(f"Nuevo weight_peak: {loss_instance.weight_peak:.4f}")

                # Reiniciar paciencia para WMAPE
                self.wait_wmape = 0
        else:
            # Si mejora WMAPE, actualizar mejor valor y reiniciar paciencia
            self.best_wmape = current_wmape
            self.wait_wmape = 0


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

# aqui es donde guarde los archivos H5 que creamos
DATA_PATH = "/storage/jhernand.2024/dataton/wl_dataton/h5_files"
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


use_float16 = False

if use_float16:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

# crea un set de entrenamiento y validacion con el 10% de los datos
train_dataset, num_train_batches = create_train_dataset_h5(os.path.join(DATA_PATH, 'train'), batch_size=8, make_ks93=False, data_fraction=0.1)
val_dataset, num_val_batches = create_val_dataset_h5(os.path.join(DATA_PATH, 'val'), batch_size=32, make_ks93=False, data_fraction=0.1)

print("train batches", num_train_batches,"val batches", num_val_batches)

model = build_unet(
            kernel_size=2,
            extend_borders=True,
            extra_layer=True,
            make_ks93=False
        )
#restart_epoch = 0

# este la unet que describimos en el documento
model_name = "hyperunet" # cambiar nombre si se cambian parametros
os.makedirs(f"/storage/jhernand.2024/dataton/wl_dataton/models/{model_name}", exist_ok=True)

log_dir = f"logs/{model_name}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)  # Crea el directorio si no existe
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
dynamic_loss_weight_callback = DynamicLossWeightScheduler(
    patience=2,     # Número de épocas sin mejora antes de ajustar los pesos
    factor=1.2,     # Incrementar los pesos en un 20% si las métricas empeoran
    min_weight=0.1, # Limitar a 0.1 el valor mínimo de los pesos
    verbose=1       # Imprimir mensajes de actualización
)

csv_logger = CSVLogger(f'/storage/jhernand.2024/dataton/wl_dataton/models/{model_name}/{model_name}_training_log.log', separator=',', append=True)

# set 0 para comenzar desde 0
restart_epoch = 0
if restart_epoch >0:
    model.load_weights(f"/storage/jhernand.2024/dataton/wl_dataton/models/{model_name}/{model_name}_{restart_epoch}.keras")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    loss=WMSELoss(), # METRICA WMSE
    metrics=[ # metricas para mostrar en el log
        ModelScore(target_wmape=2.0, target_dice=0.06, target_dpeaks=30),
        WMAPEMetric(),
        DICEEMetric(),
        DPEAKSMetric(n_peaks=3),
        tf.keras.metrics.MeanSquaredError(name="mse", dtype=None),
    ],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint( # para guardar los modelos
        filepath=f"/storage/jhernand.2024/dataton/wl_dataton/models/{model_name}/{model_name}_{{epoch}}.keras",
        save_best_only=False,
        verbose=1,
    ),
    #tf.keras.callbacks.ReduceLROnPlateau( # para ajustar learning rate. Puede ser removido
    #    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    #),
    tensorboard_callback,
    csv_logger,
]


history = model.fit(
    x=train_dataset,
    epochs=101,
    initial_epoch=restart_epoch,
    steps_per_epoch=num_train_batches,
    validation_data=val_dataset,
    validation_steps=num_val_batches,
    shuffle=False,
    validation_freq=1,
    callbacks=callbacks,
)

