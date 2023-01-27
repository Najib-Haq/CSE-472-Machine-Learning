import numpy as np

from model.Base import Base

class Softmax(Base):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def forward(self, X):
        '''
        X is a 2D array of shape (N, #classes)
        '''
        # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        exp = np.exp(X - np.max(X))
        if np.isnan(np.any(exp)):
            print("Has nan")
        return exp / np.sum(exp, axis=1, keepdims=True)   

    def backward(self, dL_dy, lr):
        return np.copy(dL_dy)