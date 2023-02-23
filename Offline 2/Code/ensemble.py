from data_handler import bagging_sampler
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from metrics import precision_score, recall_score, f1_score, accuracy
import pandas as pd

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

        # make estimators:
        self.estimators = [deepcopy(base_estimator) for _ in range(n_estimator)]

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        for i in tqdm(range(self.n_estimator)):
            print(f"\nModel {i+1}: ", end='\t')
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators[i].fit(X_sample, y_sample, show_progress=False, verbose=False)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # majority voting
        y_pred = np.sum([est.predict(X) for est in self.estimators], axis=0)
        # print(y_pred)
        # add all, if more than n_estimator/2 then majority 
        return np.where(y_pred >= (self.n_estimator/2), 1, 0)

    def get_scores(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        return {
            'Accuracy': accuracy(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }

    def compare(self, X_train, y_train, X, y):
        data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': []
        }

        data['Model'].append('Model Base')
        self.base_estimator.fit(X_train, y_train, show_progress=False, verbose=False)
        base_metrics = self.get_scores(self.base_estimator.predict(X), y)
        data['Accuracy'].append(base_metrics['Accuracy']); data['Precision'].append(base_metrics['Precision']); data['Recall'].append(base_metrics['Recall']); data['F1'].append(base_metrics['F1'])

        for i in range(self.n_estimator):
            data['Model'].append(f'Model {i+1}')
            metrics = self.get_scores(self.estimators[i].predict(X), y)
            data['Accuracy'].append(metrics['Accuracy']); data['Precision'].append(metrics['Precision']); data['Recall'].append(metrics['Recall']); data['F1'].append(metrics['F1'])
        
        data['Model'].append("Model Bagging")
        bagging_metrics = self.get_scores(self.predict(X), y)
        data['Accuracy'].append(bagging_metrics['Accuracy']); data['Precision'].append(bagging_metrics['Precision']); data['Recall'].append(bagging_metrics['Recall']); data['F1'].append(bagging_metrics['F1'])

        pd.DataFrame(data).to_csv('comparison.csv', index=False)
        

        
