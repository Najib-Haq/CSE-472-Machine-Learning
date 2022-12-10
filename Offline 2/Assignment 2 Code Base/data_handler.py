import pandas as pd
import numpy as np
import random

def load_dataset(path, label_header):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    df = pd.read_csv(path)
    print(df.head())
    X = df.drop(label_header, axis=1).values
    y = df[label_header].values
    print("Total number of samples: ", X.shape[0], "; Total number of features: ", X.shape[1])
    print("Number of positive samples: ", np.sum(y == 1), "; Number of negative samples: ", np.sum(y == 0))
    print("Positive ratio: ", np.sum(y == 1) / len(y))
    return X, y


def split_dataset(X, y, test_size, shuffle = True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    if shuffle:
        mixed = list(zip(X, y))
        random.shuffle(mixed)
        X, y = zip(*mixed)
        X, y = np.array(X), np.array(y)
    # TODO: make a version with stratified labels
    n_train_samples = int(X.shape[0] * (1 - test_size))
    X_train, y_train, X_test, y_test = X[:n_train_samples], y[:n_train_samples], X[n_train_samples:], y[n_train_samples:]
    print("Number of training samples: ", X_train.shape[0], "; Number of testing samples: ", X_test.shape[0])
    print("Number of training positive samples: ", np.sum(y_train == 1), "; Number of training negative samples: ", np.sum(y_train == 0))
    print("Number of testing positive samples: ", np.sum(y_test == 1), "; Number of testing negative samples: ", np.sum(y_test == 0))
    print("Training Positive ratio: ", np.sum(y_train== 1) / len(y_train))
    print("Testing Positive ratio: ", np.sum(y_test == 1) / len(y_test))

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    mixed = list(zip(X, y))
    mixed = random.choices(mixed, k=len(mixed))
    X_sample, y_sample = zip(*mixed)
    X_sample, y_sample = np.array(X_sample), np.array(y_sample)
    print("Training Positive ratio: ", np.sum(y_sample== 1) / len(y_sample))
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample

def set_seed(seed=25):
    """
    Set seed for random number generator
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)