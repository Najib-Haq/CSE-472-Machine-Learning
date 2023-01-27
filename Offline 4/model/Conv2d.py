import numpy as np

from model.Base import Base

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
        # kaiming initialization
        # https://paperswithcode.com/method/he-initialization
        std = np.sqrt(2 / (self.params["in_channels"] * self.params["kernel_size"] * self.params["kernel_size"]))
        # channel first shape
        kernels = np.random.randn(self.params["out_channels"], self.params["in_channels"], self.params["kernel_size"], self.params["kernel_size"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_channels"])
            return {"kernels": kernels, "bias": bias}
        return {"kernels": kernels}

    def forward(self, X):
        '''
        X shape should be (N, C, H, W)
        '''
        N, C, H, W = X.shape
        C_in, C_out, kernel_size, stride, padding, bias = self.params['in_channels'], self.params['out_channels'], self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]
        self.X = X
        # output shape
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html -> ignoring dilation
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        
        # initialize output
        output = np.zeros((N, C_out, H_out, W_out))
        # padding
        if self.params["padding"] > 0:
            self.X = np.pad(self.X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=(0, 0))
        
        # convolution
        for n in range(N): # batch
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        output[n, c, h, w] = np.sum(self.X[n, :, h*stride: h*stride + kernel_size, w*stride : w*stride + kernel_size] * self.state_dict["kernels"][c, :, :, :]) # * -> element-wise multiplication
                        if self.params["bias"]: output[n, c, h, w] += self.state_dict["bias"][c]
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
        N, _, H_out, W_out = dL_dy.shape
        C_in, C_out, kernel_size, stride, padding, bias = self.params['in_channels'], self.params['out_channels'], self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]
        H, W = self.X.shape[2], self.X.shape[3]

        # initialize gradients
        dL_dK = np.zeros((C_out, C_in, kernel_size, kernel_size)) # same as kernel shape
        dL_db = np.zeros(C_out)
        dL_dX = np.zeros((N, C_in, H, W))

        # rotate kernels 180 TODO: check this
        kernels_rotated = np.rot90(self.state_dict["kernels"], k=2, axes=(2, 3)) # (number of times rotated by 90, k=2)

        # backpropagation
        # https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
        for n in range(N): # batch
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # TODO: check this
                        dL_dK[c, :, :, :] += dL_dy[n, c, h, w] * self.X[n, :, h*stride: h*stride + kernel_size, w*stride : w*stride + kernel_size]
                        dL_dX[n, :, h*stride: h*stride + kernel_size, w*stride : w*stride + kernel_size] += dL_dy[n, c, h, w] * kernels_rotated[c, :, :, :]

                        
        if self.params["bias"]: dL_db = np.sum(dL_dy, axis=(0, 2, 3)) # channel level summation

        # update gradients
        self.grads = {"kernels": dL_dK}
        if self.params["bias"]: self.grads["bias"] = dL_db
        self.update_weights(lr)

        # remove padding
        if self.params["padding"] > 0:
            dL_dX = dL_dX[:, :, padding:-padding, padding:-padding]
        
        return dL_dX