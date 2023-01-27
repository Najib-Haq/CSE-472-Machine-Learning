import numpy as np

# write a python class from scratch for convolutional neural network
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def initialize_parameters(self):
        # randomly initialize the parameters
        self.kernels = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        if self.bias:
            self.bias = np.random.randn(self.out_channels)
        
    def forward_propagation(self, X):
        # X: (N, C, H, W)
        N, C, H, W = X.shape
        self.X = X
        # output shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        # initialize output
        self.output = np.zeros((N, self.out_channels, H_out, W_out))
        # padding
        if self.padding > 0:
            self.X = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        # convolution
        for n in range(N):
            for out_c in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        self.output[n, out_c, h, w] = np.sum(self.X[n, :, h * self.stride : h * self.stride + self.kernel_size, w * self.stride : w * self.stride + self.kernel_size] * self.kernels[out_c, :, :, :])
                        if self.bias:
                            self.output[n, out_c, h, w] += self.bias[out_c]
        return self.output

    def backward_propagation(self, X, Y):
        # X: (N, C, H, W)
        # Y: (N, out_channels, H_out, W_out)
        N, C, H, W = X.shape
        N, out_channels, H_out, W_out = Y.shape
        # initialize gradient
        self.kernels_gradient = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        if self.bias:
            self.bias_gradient = np.zeros(self.out_channels)
        # padding
        if self.padding > 0:
            self.X = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        # convolution
        for n in range(N):
            for out_c in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        self.kernels_gradient[out_c, :, :, :] += self.X[n, :, h * self.stride : h * self.stride + self.kernel_size, w * self.stride : w * self.stride + self.kernel_size] * Y[n, out_c, h, w]
                        if self.bias:
                            self.bias_gradient[out_c] += Y[n, out_c, h, w]
        return self.kernels_gradient, self.bias_gradient
