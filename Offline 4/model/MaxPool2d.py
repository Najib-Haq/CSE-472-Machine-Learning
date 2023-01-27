import numpy as np

from model.Base import Base

class MaxPool2D(Base):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.name = "MaxPool2D"
        self.params = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        self.track = {
            'max' : None,
            'shape': None,
        }

    def forward(self, X):
        N, C, H, W = X.shape
        self.track['shape'] = X.shape
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # output shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        # initialize output
        output = np.zeros((N, C, H_out, W_out))
        self.track['max'] = np.zeros((N, C, H_out, W_out))

        # max pooling
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        slice = X[n, c, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size]
                        output[n, c, h, w] = np.max(slice)
                        self.track['max'][n, c, h, w] = slice == output[n, c, h, w] # for backprop argmax

        return output

    def backward(self, dL_dy):
        N, C, H, W = dL_dy.shape
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # initialize gradient
        dL_dX = np.zeros(self.track['shape'])

        # max pooling
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        dL_dX[n, c, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size] += self.track['max'][n, c, h, w] * dL_dy[n, c, h, w]
        return dL_dX