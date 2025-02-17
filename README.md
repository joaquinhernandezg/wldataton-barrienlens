# wldataton-barrienlens


# Modulos (Directorio Scripts)
## Modulo data

Contiene funciones de i/o para leer y escribir los datos de las matrices kappa y gamma ya en formato H5. Crea batches de test y entrenamiento que luego son usados para el entrenamiento.

La función mas importante es la funcion de aumentación `make_augment_function`. Esta aplica rotaciones random en 0, 90, 180 y 270 grados a las imagenes kappa y gamma de entrenamiento. También aplica un flip un-down o left-right de manera aleatoria. Esta es la unica aumentación que se aplicó al set de datos.

## Modulo Models

Define la U-Net descrita en el documento pdf en este repo.

## Modulo Loss
Implementa los distintos tipos de loss con los que se experimentó.

### Funciones
 la funciones

- `find_n_max_pixel_positions` que encuentra las n posiciones x, y de los peaks de una imagen.
- `find_n_max_pixel_positions` que enmascara los n_edge pixeles de una imagen.
- `find_n_max_clump_positions` que suaviza una imagen con un kernel gaussiano para buscar los n máximos x, y en una imagen, ignorando los bordes.
- `find_top_k_peaks_oficial` implementación de la dataton para encontrar los peaks.
- `DPEAKS_oficial` implementación de la metrica DPEAKS de la dataton.
- `create_W` crea los pesos para WMSE
- `WMSE` implementación de la metrica WMSE
- `MAE` implementación de la metrica MAE

### Clases

- `FocalLoss` implementa la focal loss descrita en el documento
- `WMSELoss` loss que usa la metrica WMSE
- `MultiFocalLoss` Loss descrita en el documento. Incorporta distintas metricas y las pesa para crear una loss combinada.
- `WMAPELoss` Implementa la loss con metrica WMAPE
- `DICEELoss` Implementa una loss con la metrica DICEE proporcionada en la dataton
- `DPEAKSLoss` Implementa una loss con metrica DPEAKS
- `CombinedLoss` Toma una cantidad arbitraria de Losses y las combina en una unica Loss.
- `MAELoss` Implementacion de loss con metrica MAE
- `DMSLoss` Implementacion de la loss del equipo DMS.



## Modulo Metrics

Define las distintas métricas usadas para evaluar el rendimiento del modelo.
Similar al modulo loss.py


# Scripts entrenamiento
Se pueden ejecutar directamente, pero es necesario tener ya los datos en formato H5. Se recomiendan copiar directamente desde nuestro directorio en

jhernand.2024@cardano6000:/storage/jhernand.2024/dataton/wl_dataton/h5_files

Los scripts se deberían poder ejecutar directamente en el directorio correspondiente, pero es necesario ajustar los paths.

`train_normal_unet.py`: Entrena la unet de barrienlens usando la multifocal loss, con pesos ajustados dinamicamente, un reduce on plateau para ajustar el LR (opcional) entrenando con el 10% de los datos. Entrenado 100 epochs.

`train_normal_unet_wmse.py`: Entrena la unet de barrienlens usando la loss WMSE, entrenando con el 10% de los datos. Entrenado 100 epochs.


`train_normal_unet_dms_loss.py`: Entrena la unet de barrienlens usando la loss del equipo DMS, entrenando con el 10% de los datos. Entrenado 100 epochs.




