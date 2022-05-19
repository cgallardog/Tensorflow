from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def data_normalize(train_dataset, eval_dataset):
    scaler = MinMaxScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset) + len(eval_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset) + len(eval_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]
        else:
            index = i - len(train_dataset)
            all_dataset[i * samples:samples * (1 + i), :] = eval_dataset[index, :, :]

    scaler.fit(all_dataset)

    for i in range(len(train_dataset)):
        train_dataset[i, :, :] = scaler.transform(train_dataset[i, :, :])
    for i in range(len(eval_dataset)):
        eval_dataset[i, :, :] = scaler.transform(eval_dataset[i, :, :])

    return train_dataset, eval_dataset


def data_standarization(train_dataset, eval_dataset):
    scaler = StandardScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset) + len(eval_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset) + len(eval_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]
        else:
            index = i - len(train_dataset)
            all_dataset[i * samples:samples * (1 + i), :] = eval_dataset[index, :, :]

    scaler.fit(all_dataset)

    for i in range(len(train_dataset)):
        train_dataset[i, :, :] = scaler.transform(train_dataset[i, :, :])
    for i in range(len(eval_dataset)):
        eval_dataset[i, :, :] = scaler.transform(eval_dataset[i, :, :])

    return train_dataset, eval_dataset

def data_test_standarization_2(train_dataset, eval_dataset, test_dataset):
    scaler = StandardScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset) + len(eval_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset) + len(eval_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]
        else:
            index = i - len(train_dataset)
            all_dataset[i * samples:samples * (1 + i), :] = eval_dataset[index, :, :]

    scaler.fit(all_dataset)
    mean = np.mean(all_dataset, axis=0)
    stdev = np.std(all_dataset, axis=0)

    for i in range(len(test_dataset)):
        test_dataset[i, :, :] = scaler.transform(test_dataset[i, :, :])

    return test_dataset, mean[0], stdev[0]
