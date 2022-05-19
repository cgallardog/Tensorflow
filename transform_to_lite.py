import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

import Librerias.TF_Network as tf_model
import Librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import manage_dataset as main_script
import Librerias.DataTransformation as dt

model_name = 'tseq_6_q_5_neurons_128_middle_neurons_64_last_neurons_32_lr_0.01'
model_path = 'OptimizacionParametros/{}/{}.tf'.format(model_name, model_name)
save_path = 'OptimizacionParametros/{}/{}.tflite'.format(model_name, model_name)

data_path = "Dataset/zTodos/"
all_samples, training_samples, eval_samples, test_samples = dataset_samples.get_dataset()
train_dataset, eval_dataset, test_dataset = main_script.create_dataset(training_samples, eval_samples, test_samples,
                                                                       data_path)
test_dataset, mean, stdev = dt.data_test_standarization_2(train_dataset, eval_dataset, test_dataset)
testX, testY = main_script.prepare_dataset(6, 5, 1, test_dataset)

# testX, testY = shuffle(testX, testY)
num_samples = testY.shape[1]
testX = np.reshape(testX, (testX.shape[0] * testX.shape[1], 6, testX.shape[3]))
testY = np.reshape(testY, (testY.shape[0] * testY.shape[1], 1))

model = tf_model.TF_LSTM(input_shape=(testX.shape[1], testX.shape[2]), hidden_units=128, hidden_units_2=64,
                         hidden_units_3=32, layers=3, lr=0.01)
model = model.build()
model.compile(loss=tf.keras.losses.mean_squared_error,
  optimizer='adam',
  metrics=tf.keras.metrics.RootMeanSquaredError())
model.load_weights(model_path)

'''converter = tf.lite.TFLiteConverter.from_keras_model(model)
model = converter.convert()

file = open(save_path, 'wb')
file.write(model)'''

interpreter = tf.lite.Interpreter(model_path=str(save_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print(input_details["dtype"])

prediction = np.zeros((len(testX),))
for i in range(len(testX)):
    input_data = np.expand_dims(testX[i], axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    #np.append(prediction, output.argmax())
    prediction[i] = output

y_hat = prediction.flatten()
y_hat = (y_hat * stdev) + mean
y = np.array([(p * stdev) + mean for p in testY])
y = y.flatten()
rmse = mean_squared_error(y, y_hat, squared=False)

print(str(rmse))
