import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import Librerias.ConfiguracionOptimizacionParametrosForServer as param_conf
import Librerias.DataTransformation as dt
import Librerias.TF_Network as tf_model
import manage_dataset as main_script
from utils import RMSE


def evaluate_results(testX, testY, model, mean, stdev):
    a = model.evaluate(testX, testY)
    rmse_1 = stdev * model.evaluate(testX, testY)[0]
    mae_1 = stdev * model.evaluate(testX, testY)[1]

    y_hat = model.predict(testX)
    y_hat = y_hat.flatten()
    y_hat = (y_hat * stdev) + mean

    y = np.array([(p * stdev) + mean for p in testY])
    y = y.flatten()
    # x = np.transpose(y_hat)
    # y = np.transpose(testY)
    # print(len(x))
    # print(len(y))
    pearson_correlation = pearsonr(y_hat, y)
    rmse = mean_squared_error(y, y_hat, squared=False)
    mse = mean_squared_error(y, y_hat, squared=True)
    mae = mean_absolute_error(y, y_hat)

    print('RMSE:{}, MAE:{}'.format(rmse, mae))

    return rmse, mse, mae, pearson_correlation, y_hat, y


def save_and_show_results(pearson_correlation, mse, rmse, mae, model_file, conf_index, all_metrics):
    data = {
        'model_name': model_file,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson_correlation': pearson_correlation[0]
    }

    all_metrics.loc[conf_index] = data
    complete_path = dir_models + '/' + model_file + '/'

    metrics = pd.DataFrame(data, index=[0])
    metrics.pop('model_name')
    if not os.path.exists(complete_path):
        os.mkdir(complete_path)
    metrics.to_excel(complete_path + metrics_file, header=True, index=False)

    return complete_path


def plot_results(y_hat, y_original, model_path, test_samples, num_samples):
    if not os.path.exists(model_path + '/Graficas/'):
        os.mkdir(model_path + '/Graficas')
    if not os.path.exists(model_path + '/Ficheros/'):
        os.mkdir(model_path + '/Ficheros')

    tiempo_orig = np.array(list(range(0, num_samples)))
    tiempo_orig = tiempo_orig * 5 / 60

    for i in range(len(test_samples)):
        plt.figure('Profile {} '.format(test_samples[i]))
        plt.plot(tiempo_orig, y_original[i * num_samples:i * num_samples + num_samples], label='original')
        plt.plot(tiempo_orig, y_hat[i * num_samples:i * num_samples + num_samples], label='prediction')
        # plt.plot(tiempo_hat_2, y_hat, label='prediction_2')
        plt.xlabel('Time (hours)')
        plt.ylabel('Glucose (mg/dl)')
        plt.title('Profile {}'.format(test_samples[i]))
        plt.legend()

        dir_graf = model_path + '/Graficas/Y_predicha para el profile_{}.eps'.format(test_samples[i])
        plt.savefig(dir_graf, format='eps')
        # plt.show()
        plt.close()

        dir_pred = model_path + '/Ficheros/prediction_profile_{}.csv'.format(test_samples[i])
        y_original = np.reshape(y_original, (len(y_original), 1))
        y_hat = np.reshape(y_hat, (len(y_hat), 1))
        predictions = np.concatenate((y_original, y_hat), axis=1)
        pred_df = pd.DataFrame(predictions, columns=['Y_1', 'y_hat_1'])
        pred_df.to_csv(dir_pred, sep='\t', header=True, index=False)


def decide_extrapolation():
    try:
        modo_extrapolacion = int(input(""" Escriba un numero para seleccionar el modo
        1: Quiero que el trainset sea filtrado = 1\n
        2: Quiero que el trainset sea sin filtrar = 2\n
        3: Quiero que el trainset sea extrapolado = 3\n"""))
    except Exception:
        print("Error: Por favor, escriba un numero")
        modo_extrapolacion = -1

    return modo_extrapolacion


# USE_CUDA = tf.test.is_gpu_available()
# print("Is GPU available? {}".format(USE_CUDA))
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

data_path = "Dataset/zTodos/"

if not os.path.exists('OptimizacionParametros'):
    os.mkdir('OptimizacionParametros')

modo_extrapolacion = 1

# recogemos todas las muestras, ya divididas
all_samples, training_samples, eval_samples, test_samples = dataset_samples.get_dataset()
# cargamos y dividimos los dataset, juntando todos los datos en un mismo array
if modo_extrapolacion == 1:
    train_dataset, eval_dataset, test_dataset = main_script.create_dataset(training_samples, eval_samples, test_samples,
                                                                           data_path)
    dir_models = "OptimizacionParametros/Models/filtered"
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
elif modo_extrapolacion == 2:
    train_dataset, eval_dataset, test_dataset = main_script.create_extrapolated_test_dataset(training_samples,
                                                                                             eval_samples,
                                                                                             test_samples, data_path)
    dir_models = "OptimizacionParametros/Models/unfiltered"
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
else:
    data_path = "Dataset/Todos/"
    train_dataset, eval_dataset, test_dataset = main_script.create_dataset(training_samples, eval_samples,
                                                                           test_samples, data_path)
    dir_models = "OptimizacionParametros/Models/extrapolated"
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
all_models = os.listdir(path='OptimizacionParametros/GLU_3_HR_INS')

# normalizamos los datos de train y validation
# train_dataset, eval_dataset = dt.data_normalize(train_dataset, eval_dataset)
test_dataset, mean, stdev = dt.data_test_standarization_2(train_dataset, eval_dataset, test_dataset)

all_metrics = pd.DataFrame(columns=['model_name', 'MSE', 'RMSE', 'MAE', 'Pearson_correlation'])
conf_index = 0
'''for i in range(len(all_models)):
    print(all_models[i])
    if all_models[i] == 'Models':
        continue
    if all_models[i] =='GLU_3_HR_INS':
        continue
    if all_models[i].find('feat_2') != -1:
        continue'''
# cargamos todas las variables necesarias
conf_path = 'OptimizacionParametros/GLU_3_HR_INS/t_12_q_5_feat_3_neu_256_mid_neu_256_last_neu_256_l4_lr_0.001_rd_0.0/configuracion.txt'
model_params = pd.read_csv(conf_path, delim_whitespace=True)
t_seq = int(model_params['t_seq'])
H = int(model_params['H'])
q = int(model_params['q'])
n_layers = int(model_params['n_layers'])
hidden_neurons_1 = int(model_params['n_neuronas'])
hidden_neurons_2 = int(model_params['n_neuronas_2'])
last_hidden_neuron = int(model_params['n_neuronas_last'])
lr = float(model_params['LR'])

model_file = 't_12_q_5_feat_3_neu_256_mid_neu_256_last_neu_256_l4_lr_0.001_rd_0.0'
metrics_file = 'metricas_mean_sd_t_12_q_5_feat_3_neu_256_mid_neu_256_last_neu_256_l4_lr_0.001_rd_0.0.xlsx'

model_path = 'OptimizacionParametros/GLU_3_HR_INS/' + model_file + '/' + model_file + '.tf'
'''
if extrapolate == 1:
    testX, testY = prepare_dataset(t_seq, q, H, test_dataset)
else:
    testX, testY = prepare_extrapolated_dataset(t_seq, q, H, test_dataset)
'''
testX, testY = main_script.prepare_dataset(t_seq, q, H, test_dataset)

# testX, testY = shuffle(testX, testY)
num_samples = testY.shape[1]
testX = np.reshape(testX, (testX.shape[0] * testX.shape[1], t_seq, testX.shape[3]))
testY = np.reshape(testY, (testY.shape[0] * testY.shape[1], H))

model = tf_model.TF_LSTM(input_shape=(testX.shape[1], testX.shape[2]), hidden_units=hidden_neurons_1,
                         hidden_units_2=hidden_neurons_2, hidden_units_3=last_hidden_neuron,
                         layers=n_layers, lr=lr)
model = model.build()
model.compile(loss=RMSE,
              optimizer='adam',
              metrics=tf.metrics.RootMeanSquaredError())
model.load_weights(model_path)

#model = tf.keras.models.load_model('OptimizacionParametros/tseq_12_q_5_neurons_128_neurons2_0_neurons3_0_lr_0.01/2tseq_12_q_5_neurons_128_neurons2_0_neurons3_0_lr_0.01.tf')
model.summary()
# model = tf_model.TF_LSTM()
# neurons1, neurons2, neurons3, neurons4, lr, n_layers = model.get_parameters()

# lo entrenamos y testeamos
rmse, mse, mae, pearson_correlation, y_hat, y_original = evaluate_results(testX, testY, model, mean, stdev)

# params = tf_model.TF_LSTM()

result_path = save_and_show_results(pearson_correlation, mse, rmse, mae, model_file, conf_index, all_metrics)

plot_results(y_hat, y_original, result_path, test_samples, num_samples)
conf_index += 1
hypo_counter = (y_original < 70).sum()
hyper_counter = (y_original > 180).sum()
print('Hyperglycaemia:{}, Hypoglycaemia:{}'.format(hyper_counter/len(y_original), hypo_counter/len(y_original)))

all_metrics.to_excel(dir_models + '/mejores_metricas_HR_INS.xlsx', header=True, index=False)
