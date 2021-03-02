import os
import time
import math
import numpy as np
import pickle as pkl
import seaborn as sns
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.datasets import mnist, fashion_mnist

from utils.directories import *

TEST_SIZE = 20

def load_fashion_mnist(img_rows=28, img_cols=28, n_samples=None):
    print("\nLoading fashion mnist.")

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]

    num_classes = 10
    data_format = 'channels_last'

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

def preprocess_mnist(test, img_rows=28, img_cols=28, n_samples=None):
    """Preprocess mnist dataset for keras training

    :param test: If test is True, only load the first 100 images
    :param img_rows: input image n. rows
    :param img_cols: input image n. cols
    """
    print("\nLoading mnist.")

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
    else:
        if test:
            x_train = x_train[:TEST_SIZE]
            y_train = y_train[:TEST_SIZE]
            x_test = x_test[:TEST_SIZE]
            y_test = y_test[:TEST_SIZE]

    num_classes = 10
    data_format = 'channels_last'

    # # swap channels
    # x_train = np.zeros((x_train.shape[0], img_rows, img_cols, 1))
    # x_train = np.rollaxis(x_train, 3, 1)
    # x_test = np.zeros((x_test.shape[0], img_rows, img_cols, 1))
    # x_test = np.rollaxis(x_test, 3, 1)
    # data_format = "channels_first"
    # input_shape = (1, img_rows, img_cols)

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format


def _onehot(integer_labels):
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def onehot_to_labels(y):
    # if type(y) is np.ndarray:
    return np.argmax(y, axis=1)
    # elif type(y) is torch.Tensor:
    #     return torch.max(y, 1)[1]

def load_cifar(test, data, n_samples=None):
    """Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3"""
    x_train = None
    y_train = []

    data_dir=str(data)+'cifar-10/'

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "data_batch_{}".format(i))
        if i == 1:
            x_train = data_dic['data']
        else:
            x_train = np.vstack((x_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(data_dir + "test_batch")
    x_test = test_data_dic['data']
    y_test = test_data_dic['labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4)
    y_train = np.array(y_train)

    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4)
    y_test = np.array(y_test)

    input_shape = x_train.shape[1:]
    num_classes = 10
    data_format = 'channels_first'

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
    else:
        if test:
            x_train = x_train[:TEST_SIZE]
            y_train = y_train[:TEST_SIZE]
            x_test = x_test[:TEST_SIZE]
            y_test = y_test[:TEST_SIZE]

    return x_train, _onehot(y_train), x_test, _onehot(y_test), input_shape, num_classes, data_format


def load_dataset(dataset_name, test, data=DATA_PATH, n_samples=None):
    """
    Load dataset.
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: If True only loads the first 100 samples
    """
    # global x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

    if dataset_name == "mnist":
        return preprocess_mnist(test=test, n_samples=n_samples)
    elif dataset_name == "cifar":
        return load_cifar(test=test, data=data, n_samples=n_samples)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(n_samples=n_samples)
    else:
        raise ValueError("\nWrong dataset name.")


def save_to_pickle(data, relative_path, filename):
    """ saves data to pickle """

    filepath = relative_path + filename
    print("\nSaving pickle: ", filepath)
    os.makedirs(filepath, exist_ok=True)
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)


def unpickle(file):
    """ Load byte data from file"""
    with open(file, 'rb') as f:
        data = pkl.load(f, encoding='latin-1')
    return data


def load_from_pickle(path, test=False):
    """ loads data from pickle containing: x_test, y_test."""
    print("\nLoading from pickle: ",path)
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    if test is True:
        data = data[:TEST_SIZE]
    return data
