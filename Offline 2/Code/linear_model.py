import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.n_features = params['n_features']
        self.weights = np.random.rand(self.n_features) if params['random_init'] else np.zeros(self.n_features)
        self.bias = np.random.rand() if params['random_init'] else 0
        self.lr = params['learning_rate']
        self.iter = params['iterations']
        self.threshold = params['threshold']
        self.min = 0
        self.max = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def loss(self, y_true, y_pred, n_samples):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        return (-1 / n_samples) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y, show_progress=True, verbose=False):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # print("Initial params : ", self.weights, self.bias)
        n_samples = X.shape[0]
        # X = self.data_scale(X, is_train=True)
        dataloader = tqdm(range(self.iter)) if show_progress else range(self.iter)
        for i in dataloader:
            # Andrew NG: https://www.youtube.com/watch?v=t1IT5hZfS48&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=2
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            # here X.T as need for every single weight
            # X.T shape (n_features, n_samples) ; y_pred and y shape (n_samples, 1) -> this gets broadcasted
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) 
            # same formula here x = 1 so no need dot
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if(verbose and i % 100 == 0):
                print("Loss at iter ", i, " : ", self.loss(y, y_pred, n_samples))

        # print("Final params : ", self.weights, self.bias)


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # X = self.data_scale(X, is_train=False)
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.where(y_pred >= self.threshold, 1, 0)
