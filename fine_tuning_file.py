import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.interpolate import interp1d
import os

import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import Librerias.TuningParametrosServer as param_conf
import Librerias.DataTransformation as dt
import Librerias.TF_Network as tf_model
from utils import RMSE
import manage_dataset


def train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=0, hidden_neurons_2=0, last_hidden_neuron=0,
                   lr=0.01, n_layers=1, extrapolated=1):
    model_file = 'tseq_{}_q_{}_feat_{}_neurons_{}_middle_neurons_{}_last_neurons_{}_l{}_lr_{}'.format(t_seq, q,
                                                                                                      trainX.shape[-1],
                                                                                                      hidden_neurons_1,
                                                                                                      hidden_neurons_2,
                                                                                                      last_hidden_neuron,
                                                                                                      n_layers, lr)
    model_path = 'OptimizacionParametros/' + model_file + '/'
    checkpoint_path = model_path + model_file + '.tf'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True)
    tensorboard_checkpoint = tf.keras.callbacks.TensorBoard('logs/scalars/{}'.format(model_file), update_freq=1)
    stop_fit_early = tf.keras.callbacks.EarlyStopping(patience=10)

    history = model.fit(trainX, trainY, validation_split=0.25, shuffle=True, epochs=100,
                        batch_size=32, verbose=1, use_multiprocessing=True, workers=5, callbacks=[model_checkpoint,
                                                                                                  tensorboard_checkpoint,
                                                                                                  stop_fit_early])

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    save_model_path = model_path + '/' + model_file + '.tf'
    model.save_weights(save_model_path, save_format='tf')

    return model_path


def model_builder(hp):
    min_neuron1, max_neuron1, step1 = parameters.get_neuronas_1()
    min_neuron2, max_neuron2, step2 = parameters.get_neuronas_2()
    min_neuron3, max_neuron3, step3 = parameters.get_neuronas_3()
    '''min_neuron2, step2, step3 = parameters.get_fixed_neurons()'''
    min_layers, max_layers = parameters.get_combinaciones_n_layers()
    lr = parameters.get_combinaciones_lr()
    dropout, recurrent_dropout = parameters.get_dropout()


    # hp_units = hp.HParam('num_units', hp.Discrete([8, 16, 32, 64, 128, 256])
    model = tf.keras.Sequential()
    layer = hp.Int('n_layers', min_value=min_layers, max_value=max_layers, step=1)
    '''hp_units = hp.Int('units', min_value=min_neuron1, max_value=max_neuron1, step=step1)'''
    hp_units = hp.Choice('units', values=step1)
    if max_layers > 0:
        hp_units_3 = hp.Choice('units3', values=step3)
    elif max_layers > 1:
        hp_units_2 = hp.Choice('units2', values=step2)
    model.add(tf.keras.layers.LSTM(units=hp_units, input_shape=(trainX.shape[1], trainX.shape[2]),
                                   return_sequences=layer > 1))  # LSTM espera [samples, timesteps, features]
    if layer + 1 > 1:
        if layer + 1 > 2:
            for _ in range(layer - 2):
                hp_dropout = hp.Choice('dropout', values=dropout)
                hp_recurrent_dropout = hp.Choice('recurrent_dropout', values=recurrent_dropout)
                '''hp_units_2 = hp.Int('units2', min_value=min_neuron2, max_value=max_neuron2, step=step2)'''
                model.add(tf.keras.layers.LSTM(units=hp_units_2, return_sequences=True, dropout=hp_dropout,
                                               recurrent_dropout=hp_recurrent_dropout))
        '''hp_units_3 = hp.Int('units3', min_value=min_neuron3, max_value=max_neuron3, step=step3)'''
        model.add(tf.keras.layers.LSTM(units=hp_units_3, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_learning_rate = hp.Choice('learning_rate', values=lr)

    model.compile(loss=RMSE,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError()])

    return model


def decide_extrapolation():
    try:
        modo_extrapolacion = int(input(""" Escriba un numero para seleccionar el modo
        1: Quiero que el trainset sea filtrado = 1\n
        2: Quiero que el trainset sea sin filtrar = 2\n
        3: Quiero que el trainset sea extrapolado = 3\n"""))
    except Exception:
        print("Error: Por favor, escriba un numero")
        modo_extrapolacion = -1

    try:
        modo_manual = int(input(""" Escriba un numero para seleccionar el modo
        1: Quiero usar fine tuning = 1\n
        2: Quiero realizar un entrenamiento manual = 2\n"""))
    except Exception:
        print("Error: Por favor, escriba un numero")
        modo_manual = -1

    return modo_extrapolacion, modo_manual


def prepare_extrapolated_dataset(t_seq, q, H, dataset):
    # preparamos el dataset para crear una sliding window
    len_file = len(dataset[0]) - t_seq - q - H
    dataX2, dataY2 = [], []  # shape = (samples, file_length, prev_points=timesteps, features)
    for index in range(len(dataset)):  # dataset
        dataX = []
        dataY = np.empty(shape=(len_file, H))
        for i in range(200, len_file):
            a = dataset[index, i:(i + t_seq), :]
            b = dataset[index, i + t_seq + q, 0]
            extrap_data = pd.DataFrame(a)
            if not extrap_data[0].all():
                ceros = (extrap_data[0] == 0).sum()
                x = range(0, len(extrap_data[0]) - 2)
                f = interp1d(x, extrap_data[0][:3], fill_value="extrapolate")
                ma = f([4, 5])
            extrap_data = extrap_data.to_numpy()
            dataX.append(extrap_data)
            dataY[i, 0] = b
        dataX2.append(dataX)
        dataY2.append(dataY)
    return np.array(dataX2), np.array(dataY2)


def save_configuration(t_seq, H, q, n_layers, n_neuronas, n_neuronas_2, n_neuronas_last, lr, model_path, configuration, scaler):
    data_config = {
        't_seq': t_seq,
        'H': H,
        'q': q,
        'n_layers': n_layers,
        'n_neuronas': n_neuronas,
        'n_neuronas_2': n_neuronas_2,
        'n_neuronas_last': n_neuronas_last,
        'optimizer_alg': 1,
        'name_optimizer': 'Adam',
        'LR': lr,
    }
    configuration.loc[0] = data_config
    configuration.to_csv(model_path + '/configuracion.txt', header=True, index=False, sep='\t')
    configuration.drop(0, axis=0)


# USE_CUDA = tf.test.is_gpu_available()
# print("Is GPU available? {}".format(USE_CUDA))
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

data_path = "Dataset/zTodos/"

if not os.path.exists('OptimizacionParametros'):
    os.mkdir('OptimizacionParametros')

# recogemos todas las muestras, ya divididas
all_samples, training_samples, eval_samples, test_samples = dataset_samples.get_dataset()
# cargamos y dividimos los dataset, juntando todos los datos en un mismo array
train_dataset, eval_dataset, test_dataset = manage_dataset.create_dataset(training_samples, eval_samples, test_samples,
                                                                          data_path)

# cargamos todas las variables necesarias
parameters = param_conf.CombinacionParametrosRedOptimizar()
all_t_seq = parameters.get_combinaciones_tseq()  # numero de puntos previos que usaremos
all_q = parameters.get_combinaciones_q()  # horizonte de predicción (que valor futuro quiero predecir)
H = 1

# normalizamos los datos de train y validation
# train_dataset, eval_dataset = dt.data_normalize(train_dataset, eval_dataset)
train_dataset, eval_dataset, scaler = dt.data_standarization(train_dataset, eval_dataset)

configuration = pd.DataFrame(columns=['t_seq', 'H', 'q', 'n_layers', 'n_neuronas', 'n_neuronas_2', 'n_neuronas_last',
                                      'optimizer_alg', 'name_optimizer', 'LR'])
for t_seq in all_t_seq:
    for q in all_q:
        trainX, trainY = manage_dataset.prepare_dataset(t_seq, q, H, train_dataset)
        # Adaptamos los set de datos para que puedan entrar en la LSTM
        trainX = np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], t_seq, trainX.shape[3]))
        trainY = np.reshape(trainY, (trainY.shape[0] * trainY.shape[1], H))
        # Mezclamos las muestras del training y validation set
        trainX, trainY = shuffle(trainX, trainY)

        valX, valY = manage_dataset.prepare_dataset(t_seq, q, H, eval_dataset)
        valX, valY = shuffle(valX, valY)
        valX = np.reshape(valX, (valX.shape[0] * valX.shape[1], t_seq, valX.shape[3]))
        valY = np.reshape(valY, (valY.shape[0] * valY.shape[1], H))

        # creamos el modelo
        turner = kt.Hyperband(model_builder, objective='val_loss', max_epochs=50, factor=2, directory='my_dir_3',
                              project_name='kt_hyperparameters_3')
        '''turner = kt.BayesianOptimization(model_builder, objective='val_loss')'''
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        turner.search(trainX, trainY, validation_split=0.25, shuffle=True, epochs=100, use_multiprocessing=True,
                      workers=10, callbacks=[stop_early, tf.keras.callbacks.TensorBoard('tune_4', update_freq=1)])
        best_hps = turner.get_best_hyperparameters(num_trials=1)[0]
        model = turner.hypermodel.build(best_hps)
        '''_, max_layers = parameters.get_combinaciones_n_layers()'''
        _, max_layers = parameters.get_combinaciones_n_layers()
        hidden_units = best_hps['units']
        if max_layers < 3:
            hidden_units_2 = 0
            if max_layers < 2:
                hidden_units_3 = 0
            else:
                hidden_units_3 = best_hps['units3']
        else:
            hidden_units_2 = best_hps['units2']
            hidden_units_3 = best_hps['units3']
        lr = best_hps['learning_rate']
        n_layers = best_hps['n_layers']

        # lo entrenamos y testeamos
        model_path = train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=hidden_units,
                                    hidden_neurons_2=hidden_units_2, last_hidden_neuron=hidden_units_3, lr=lr,
                                    n_layers=n_layers)
        save_configuration(t_seq, H, q, n_layers, hidden_units, hidden_units_2, hidden_units_3, lr, model_path,
                           configuration, scaler)
