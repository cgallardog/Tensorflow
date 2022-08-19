import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os
from ClarkeErrorGrid import zone_percentages, clarke_error_grid

import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import Librerias.ConfiguracionOptimizacionParametrosForServer as param_conf
import Librerias.DataTransformation as dt
import Librerias.TF_Network as tf_model
from utils import RMSE
import manage_dataset

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
'''train_dataset, eval_dataset, scaler = dt.data_standarization(train_dataset, eval_dataset)
test_dataset, mean, stdev = dt.data_test_standarization_2(train_dataset, eval_dataset, test_dataset)'''

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

        model = SVR(verbose=True)
        model.fit(trainX[:, :, 0], trainY)

        testX, testY = manage_dataset.prepare_dataset(t_seq, q, H, test_dataset)
        # testX, testY = shuffle(testX, testY)
        num_samples = testY.shape[1]
        testX = np.reshape(testX, (testX.shape[0] * testX.shape[1], t_seq, testX.shape[3]))
        testY = np.reshape(testY, (testY.shape[0] * testY.shape[1], H))

        y_hat = model.predict(testX[:, :, 0])
        y_hat = y_hat.flatten()
        '''y_hat = (y_hat * stdev) + mean

        y = np.array([(p * stdev) + mean for p in testY])'''
        y = testY
        y = y.flatten()
        pearson_correlation = pearsonr(y_hat, y)
        rmse = mean_squared_error(y, y_hat, squared=False)
        mse = mean_squared_error(y, y_hat, squared=True)
        mae = mean_absolute_error(y, y_hat)

        print('RMSE:{}, MAE:{}, Pearson Correlation:{}'.format(rmse, mae, pearson_correlation))

        y_original = np.reshape(y, (len(y), 1))
        y_hat = np.reshape(y_hat, (len(y_hat), 1))
        predictions = np.concatenate((y_original, y_hat), axis=1)
        pred_df = pd.DataFrame(predictions, columns=['Y_1', 'y_hat_1'])

        y = pred_df['Y_1']
        y_hat = pred_df['y_hat_1']
        plot, zone = clarke_error_grid(y, y_hat, 'tseq_{}_H_1_q_{}'.format(t_seq, q))
        total_percentages = zone_percentages('SVR', zone)
        all_percentages = pd.DataFrame(columns=['Model', 'A_zone', 'B_zone', 'C_zone', 'D_zone', 'E_zone'])
        all_percentages = pd.concat([all_percentages, total_percentages])
        all_percentages.head(1)
        print('ya')
