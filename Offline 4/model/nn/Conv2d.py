import numpy as np

from model.nn.Base import Base

class Conv2D(Base):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.name = "Conv2D"
        self.params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }
        self.state_dict = self.initialize_parameters()

    def initialize_parameters(self):
        # Xavier initialization
        std = np.sqrt(2 / (self.params["in_channels"] * self.params["kernel_size"] * self.params["kernel_size"]))
        # channel first shape
        kernels = np.random.randn(self.params["out_channels"], self.params["in_channels"], self.params["kernel_size"], self.params["kernel_size"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_channels"])
            return {"kernels": kernels, "bias": bias}
        return {"kernels": kernels}

    def generate_strided_tensor(self, X, kernel_size, stride, padding, out_shape):
        '''
        here kernel_size, stride, padding are tuples of (H, W)
        ''' 
        C_out = X.shape[1]
        N, _, H_out, W_out = out_shape

        # pad the input tensor if necessary
        if padding != (0, 0):
            X = np.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode="constant", constant_values=0)
        
        # get strides of X
        N_strides, C_out_strides, H_strides, W_strides = X.strides
        # create a strided tensor
        # use this link to understand: https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        strided_tensor = np.lib.stride_tricks.as_strided(
            X, 
            shape=(N, C_out, H_out, W_out, kernel_size[0], kernel_size[1]), 
            strides=(N_strides, C_out_strides, stride[0] * H_strides, stride[1] * W_strides, H_strides, W_strides)
        )
        return strided_tensor


    def forward(self, X):
        '''
        X shape should be (N, C, H, W)
        '''

        kernel_size, stride, padding, bias = self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]
        # get output shape
        B, C_in, H_in, W_in = X.shape
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        output_shape = (B, C_in, H_out, W_out)

        # get strided X windows
        strided_X = self.generate_strided_tensor(X, (kernel_size, kernel_size), (stride, stride), (padding, padding), output_shape)

        # convolution with kernels
        # use this to understand: https://ajcr.net/Basic-guide-to-einsum/
        output = np.einsum("nchwkl,ockl->nohw", strided_X, self.state_dict["kernels"])

        # add bias if necessary
        if bias:
            output += self.state_dict["bias"][np.newaxis, :, np.newaxis, np.newaxis]  
        
        self.cache = {
            "strided_X": strided_X,
            "X_shape": X.shape
        }

        return output

    def backward(self, dL_dy, lr):
        '''
        dL_dy = gradient of the cost with respect to the output of the conv layer -> (bs, C_out, H, W)

        compute :
        dL_dK = gradient of the cost with respect to the kernels -> (C_out, C_in, kernel_size, kernel_size)
        dL_db = gradient of the cost with respect to the bias -> (C_out)
        dL_dX = gradient of the cost with respect to the input -> (bs, C_in, H_in, W_in)

        '''
        # get parameters
        kernel_size, stride, padding, bias = self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]

        # compute dL_dK and dL_db
        dL_dK = np.einsum("nchwkl,nohw->ockl", self.cache['strided_X'], dL_dy) # Convolution(X, dL_dy)
        if bias:
            dL_db = np.einsum('nohw->o', dL_dy) # sum over N, H, W

        # compute dL_dX
        # rotate kernels 180
        kernels_rotated = np.rot90(self.state_dict["kernels"], k=2, axes=(2, 3)) # (number of times rotated by 90, k=2)
        # get strided dL_dy windows
        dout_padding = kernel_size - 1 if padding == 0 else kernel_size - 1 - padding
        dout_dilate = stride - 1
        # dilate dL_dy based on stride
        if dout_dilate != 0:
            dL_dy_dilated = np.insert(dL_dy, range(1, dL_dy.shape[2]), values=0, axis=2) # args - input, index, value, axis
            dL_dy_dilated = np.insert(dL_dy_dilated, range(1, dL_dy.shape[3]), values=0, axis=3)
        else:
            dL_dy_dilated = dL_dy.copy()
        # TODO: check if stride is original or this one
        strided_dL_dy = self.generate_strided_tensor(dL_dy_dilated, (kernel_size, kernel_size), (1, 1), (dout_padding, dout_padding), self.cache['X_shape'])
        
        # compute dL_dX
        dL_dX = np.einsum("nohwkl,ockl->nchw", strided_dL_dy, kernels_rotated) # Convolution(padded dL_dy, kernels_rotated)

        # update parameters
        self.grads['kernels'] = dL_dK
        if bias: self.grads['bias'] = dL_db
        self.update_weights(lr)

        return dL_dX