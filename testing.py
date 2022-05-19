import tensorflow as tf
import numpy as np
import data
from sklearn.metrics import mean_squared_error

import Librerias.TF_Network as tf_model
from utils import RMSE

subjects = [540, 544, 552, 567, 584, 596]


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
print("loading data..")
X_train, Y_train, X_val, Y_val, X_train_mean, X_train_stdev = data.load_data(dir="Dataset/raw",
            history_length=6, prediction_horizon=6,
            include_missing=True, train_frac=0.8)
print("data loaded.\nmean = {0}\tstdev = {1}".format(round(X_train_mean, 3), round(X_train_stdev, 3)))


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

LSTM_model = tf_model.TF_LSTM()
for ridx in range(2):
    model = LSTM_model.build()
    #neurons1, neurons2, neurons3, neurons4, lr, n_layers = model.get_parameters()
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=tf.keras.metrics.RootMeanSquaredError())

    callback = []
    callback.append(tf.keras.callbacks.ModelCheckpoint(filepath='{0}/{1}_{2}.ckpt'.format('OptimizacionParametros/Models', str(model), ridx+1),
                                         save_best_only=True, save_weights_only=True))
    callback.append(tf.keras.callbacks.EarlyStopping(patience=30))

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), shuffle=True, epochs=300,
                        batch_size=32, verbose=1, use_multiprocessing=True, workers=5, callbacks=callback)


    train_loss = X_train_stdev * model.evaluate(X_train, Y_train)[1]
    val_loss = X_train_stdev * model.evaluate(X_val, Y_val)[1]

    print("training RMSE = {0}".format(train_loss))
    print("validation RMSE = {0}".format(val_loss))

    for subject_index in subjects:
        X_test, Y_test, test_times = data.load_test_data(dir="Dataset/raw", subject_index=subject_index,
                                                         history_length=6,
                                                         prediction_horizon=6,
                                                         data_mean=X_train_mean, data_stdev=X_train_stdev)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        aa = model.predict(X_test)
        rmse_2 = X_train_stdev*mean_squared_error(Y_test, aa, squared=False)
        # evaluate model
        rmse = X_train_stdev * np.sqrt(model.evaluate(X_test, Y_test)[1])
        rmse = X_train_stdev*model.evaluate(X_test, Y_test)[1]
        rmse_3 = 0
        for i in range(len(aa)):
            rmse_3 = rmse_3 + (aa[i] - Y_test[i])**2
        rmse_3 = rmse_3/len(aa)
        rmse_3 = X_train_stdev*np.sqrt(rmse_3)

        preds = [(p[0] * X_train_stdev) + X_train_mean for p in model.predict(X_test)]

        print('RMSE:{}, RMSE_MALA:{}'.format(rmse, rmse_2))

