import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt

import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import Librerias.ConfiguracionOptimizacionParametrosForServer as param_conf
import Librerias.DataTransformation as dt
import Librerias.TF_Network as tf_model
from utils import RMSE
import manage_dataset


def train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=0, hidden_neurons_2=0, last_hidden_neuron=0,
                   lr=0.01, n_layers=1, recurrent_dropout=0.0, extrapolated=1):
    model_file = \
        't_{}_q_{}_feat_{}_neu_{}_mid_neu_{}_last_neu_{}_l{}_lr_{}_rd_{}'.format(t_seq, q, trainX.shape[-1],
                                                                                 hidden_neurons_1, hidden_neurons_2,
                                                                                 last_hidden_neuron, n_layers, lr,
                                                                                 recurrent_dropout)
    model_path = 'OptimizacionParametros/GLU_3_Ohio/' + model_file + '/'
    checkpoint_path = model_path + model_file + '.tf'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True)
    tensorboard_checkpoint = tf.keras.callbacks.TensorBoard('GLU_3_Ohio/scalars/{}'.format(model_file), update_freq=1)
    stop_fit_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

    history = model.fit(trainX, trainY, validation_split=0.25, shuffle=True, epochs=200,
                        batch_size=32, verbose=1, use_multiprocessing=True, workers=5, callbacks=[model_checkpoint,
                                                                                                  tensorboard_checkpoint])

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    save_model_path = model_path + '/' + model_file + '.tf'
    model.save_weights(save_model_path, save_format='tf')

    return model_path, history


def save_configuration(t_seq, H, q, n_layers, n_neuronas, n_neuronas_2, n_neuronas_last, lr, model_path, configuration, scaler=None):
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

if not os.path.exists('OptimizacionParametros/GLU_3_Ohio'):
    os.mkdir('OptimizacionParametros/')

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
train_dataset = dt.data_normalization(train_dataset)    #### OJO: Especificar los máximos ####
#train_dataset, eval_dataset, scaler = dt.data_standarization(train_dataset, eval_dataset)

configuration = pd.DataFrame(columns=['t_seq', 'H', 'q', 'n_layers', 'n_neuronas', 'n_neuronas_2', 'n_neuronas_last',
                                      'optimizer_alg', 'name_optimizer', 'LR'])
for t_seq in all_t_seq:
    for q in all_q:
        # prepare, reshape and shuffle the dataset
        trainX, trainY = manage_dataset.dataset(t_seq, q, H, train_dataset)
        valX, valY = manage_dataset.dataset(t_seq, q, H, eval_dataset)

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
                        model.summary()

                        # lo entrenamos y testeamos
                        model_path, history = train_and_save(trainX, trainY, model, t_seq, q, hidden_neurons_1=hidden_units,
                                                    hidden_neurons_2=hidden_units_2, last_hidden_neuron=hidden_units_3,
                                                    lr=lr, n_layers=layer, recurrent_dropout=recurrent_dropout)

                        # representamos la pérdida a lo largo del entrenamiento en una gráfica
                        epochs_ran = len(history.history['loss'])
                        plt.plot(range(0, epochs_ran), history.history['val_root_mean_squared_error'],
                                 label='Validation')
                        plt.plot(range(0, epochs_ran), history.history['root_mean_squared_error'], label='Training')
                        plt.legend()
                        plt.savefig(model_path + 'Loss_Plot', format='eps')
                        #plt.show()

                        '''save_configuration(t_seq, H, q, layer, hidden_units, hidden_units_2, hidden_units_3, lr, 
                                           model_path, configuration)'''
                        save_configuration(t_seq, H, q, layer, hidden_units, hidden_units_2, hidden_units_3, lr,
                                           model_path, configuration)
