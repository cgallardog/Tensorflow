import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.interpolate import interp1d
import os

import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import Librerias.ConfiguracionOptimizacionParametrosForServer as param_conf
import Librerias.DataTransformation as dt
import Librerias.TF_Network as tf_model
from utils import RMSE
import manage_dataset


def train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=0, hidden_neurons_2=0, last_hidden_neuron=0,
                   lr=0.01, n_layers=1, recurrent_dropout=0.0, extrapolated=1):
    if n_layers == 1:
        model_file = 'tseq_{}_q_{}_neurons_{}_l{}_lr_{}_INS_ING'.format(t_seq, q, hidden_neurons_1, n_layers, lr)
    elif n_layers == 2:
        model_file = 'tseq_{}_q_{}_neurons_{}_last_neurons_{}_l{}_lr_{}'.format(t_seq, q, hidden_neurons_1,
                                                                                last_hidden_neuron, n_layers, lr)
    else:
        model_file = \
            't_{}_q_{}_feat_{}_neu_{}_mid_neu_{}_last_neu_{}_l{}_lr_{}_rd_{}'.format(t_seq, q, trainX.shape[-1],
                                                                                     hidden_neurons_1, hidden_neurons_2,
                                                                                     last_hidden_neuron, n_layers, lr,
                                                                                     recurrent_dropout)
    model_path = 'OptimizacionParametros/GLU_HR' + model_file + '/'
    checkpoint_path = model_path + model_file + '.tf'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True)
    tensorboard_checkpoint = tf.keras.callbacks.TensorBoard('GLU_HR/scalars/{}'.format(model_file), update_freq=1)
    stop_fit_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(trainX, trainY, validation_split=0.25, shuffle=True, epochs=100,
                        batch_size=32, verbose=1, use_multiprocessing=True, workers=5, callbacks=[model_checkpoint,
                                                                                                  tensorboard_checkpoint,
                                                                                                  stop_fit_early])

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    save_model_path = model_path + '/' + model_file + '.tf'
    model.save_weights(save_model_path, save_format='tf')

    return model_path


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
        'LR': lr
    }
    configuration.loc[0] = data_config
    configuration.to_csv(model_path + '/configuracion.txt', header=True, index=False, sep='\t')
    configuration.drop(0, axis=0)


# USE_CUDA = tf.test.is_gpu_available()
# print("Is GPU available? {}".format(USE_CUDA))
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

data_path = "Dataset/zTodos/"

if not os.path.exists('OptimizacionParametros/GLU_HR'):
    os.mkdir('OptimizacionParametros/GLU_HR')

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

        n_layers = parameters.get_combinaciones_n_layers()
        n1, n2, n3 = parameters.get_fixed_neurons()
        lr = parameters.get_fixed_lr()
        dropout, recurrent_dropout = parameters.get_dropout()
        for layer in n_layers:
            for hidden_units in n1:
                for hidden_units_2 in n2:
                    for hidden_units_3 in n3:
                        LSTM_model = tf_model.TF_LSTM(input_shape=(trainX.shape[1], trainX.shape[2]), hidden_units=hidden_units,
                                                      hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3,
                                                      layers=layer, dropout=dropout, recurrent_dropout=recurrent_dropout)
                        model = LSTM_model.build()
                        # neurons1, neurons2, neurons3, neurons4, lr, n_layers = model.get_parameters()
                        model.compile(loss=RMSE,
                                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                               tf.keras.metrics.MeanAbsoluteError()])

                        # lo entrenamos y testeamos
                        model_path = train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=hidden_units,
                                                    hidden_neurons_2=hidden_units_2, last_hidden_neuron=hidden_units_3, lr=lr,
                                                    n_layers=layer, recurrent_dropout=recurrent_dropout)
                        '''save_configuration(t_seq, H, q, layer, hidden_units, hidden_units_2, hidden_units_3, lr, model_path,
                                           configuration)'''
                        save_configuration(t_seq, H, q, layer, hidden_units, hidden_units_2, hidden_units_3, lr, model_path,
                                           configuration, scaler)

