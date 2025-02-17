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




# Resultados Experimentos

Resultados experimentos comentados en discord. Todos ellos son con el 10% de los datos
y 100 epochs de entrenamiento.


## UNET - WMSE LOSS - ReduceLROnPlateau
100 epochs con el 10% de los datos


WMAPE: 2.1631603240966797 +- 1.81206476688385
WMSE: 7.543821266153827e-05 +- 5.136361869517714e-05
WMAE: 0.006721491925418377 +- 0.0023870253935456276
MAE: 0.006653583608567715 +- 0.002408441388979554
DICEE: 0.0750947892665863 +- 0.007932317443192005
DPEAKS: 37.39514923095703 +- 70.16960144042969
L2DPEAKS: 13.180429458618164 +- 29.755455017089844
Hong15: 0.3308532238006592 +- 0.21917925775051117

## UNET - WMSE LOSS

WMAPE: 2.044210910797119 +- 1.711029291152954
WMSE: 6.666579429293051e-05 +- 4.71141429443378e-05
WMAE: 0.0063722701743245125 +- 0.002313920296728611
MAE: 0.006328224204480648 +- 0.0023343353532254696
DICEE: 0.07038053870201111 +- 0.00814793910831213
DPEAKS: 35.91340637207031 +- 69.23721313476562
L2DPEAKS: 12.220766067504883 +- 28.54241943359375
Hong15: 0.2885902523994446 +- 0.18613924086093903

## UNET - DMS Loss - ReduceLROnPlateau
100 epochs con el 10% de los datos

parametros por default
eta_1=1 eta_2=1e-3

WMAPE: 2.123208522796631 +- 1.8383846282958984
WMSE: 7.132572500267997e-05 +- 5.836071795783937e-05
WMAE: 0.006517808884382248 +- 0.0025685240980237722
MAE: 0.00644917506724596 +- 0.0025739152915775776
DICEE: 0.07117462158203125 +- 0.007543256971985102
DPEAKS: 34.55886459350586 +- 68.12926483154297
L2DPEAKS: 11.919792175292969 +- 28.254365921020508
Hong15: 0.31956756114959717 +- 0.24052222073078156


## UNET - MULTIFOCAL LOSS - DYNAMIC WEIGHT SCHEDULER - ReduceLROnPlateau
100 epochs con el 10% de los datos

Output del metric script
WMAPE: 1.8617124557495117 +- 1.66322922706604
WMSE: 5.5522141337860376e-05 +- 4.867161260335706e-05
WMAE: 0.00579866673797369 +- 0.0025831228122115135
MAE: 0.005752179306000471 +- 0.0025999085046350956
DICEE: 0.05658911541104317 +- 0.0057911803014576435
DPEAKS: 26.409561157226562 +- 61.14170837402344
L2DPEAKS: 8.940558433532715 +- 24.871572494506836
Hong15: 0.24898409843444824 +- 0.16344450414180756


