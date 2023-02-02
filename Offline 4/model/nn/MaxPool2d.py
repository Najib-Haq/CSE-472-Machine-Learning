import numpy as np
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from nn.Base import Base

class MaxPool2D(Base):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.name = "MaxPool2D"
        self.params = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        self.cache = {}

    def forward(self, X):
        N, C, H, W = X.shape
        self.cache['X_shape'] = X.shape
        self.cache['X_strides'] = X.strides
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # output shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        # get kernel strided X
        N_strides, C_out_strides, H_strides, W_strides = X.strides
        strided_X = np.lib.stride_tricks.as_strided(
            X,
            shape=(N, C, H_out, W_out, kernel_size, kernel_size),
            strides=(N_strides, C_out_strides, stride * H_strides, stride * W_strides, H_strides, W_strides)
        )

        # max pooling
        output = np.max(strided_X, axis=(4, 5))
        print("X Strides: ", X.strides)
        self.cache['strided_X'] = strided_X
        return output

    def backward(self, dL_dy):
        N, C, H_out, W_out = dL_dy.shape
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # get cached strided_X
        strided_X = self.cache['strided_X']
        reshaped_strided_X = strided_X.reshape(N, C, H_out, W_out, -1) # need to do this as cannot get max from multiple axis
        argmaxes = reshaped_strided_X.argmax(axis=-1)
        a1, a2, a3, a4 = np.indices((N, C, H_out, W_out)) # indices of axies
        argmaxes_indices = (a1, a2, a3, a4, argmaxes)

        # set to 1 and then multiply with gradient
        strided_X_maxes = np.zeros_like(reshaped_strided_X)
        strided_X_maxes[argmaxes_indices] = 1
        strided_X_maxes *= dL_dy[..., None]

        # reshape to original shape
        strided_X_maxes = strided_X_maxes.reshape(strided_X.shape)
        print("Strided X Maxes Strides: ", strided_X_maxes.strides)

        # now go back to original, hopefully this maps back correctly
        N_strides, C_out_strides, H_strides, W_strides = self.cache['X_strides']
        dL_dX = np.lib.stride_tricks.as_strided(
            strided_X_maxes,
            shape=self.cache['X_shape'],
            strides = (72, 32, 16, 8),
            # strides=(N_strides, C_out_strides, stride * H_strides, stride * W_strides)
        )
        
        print("dL_dX : ", dL_dX)
        return dL_dX


if __name__ == "__main__":
    np.random.seed(42)
    mp = MaxPool2D(2, 1)
    X = np.random.rand(1,1,3,3)
    out = mp.forward(X)
    print("X: ", X)
    print("Forward: ", out.shape)
    dL_dy = np.random.rand(1,1,2,2)
    print("dL_dy: ", dL_dy)
    dL_dX = mp.backward(dL_dy)
    print("Backward: ", dL_dX.shape)