import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.Conv2d import Conv2D
np.random.seed(42)

def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0, back=False):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))
        # if back: working_input = working_input[:, :, 1:-1, 1:-1]
    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    print("Working input shape: ", working_input)
    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )


class Conv2D2:
    """
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters.
        """

        n, c, h, w = x.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        # add bias to kernels
        out += self.bias[None, :, None, None]

        self.cache = x, windows
        print(self.weight[0, 0, :, :])

        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: dx, dw, and db relative to this module
        """
        
        # print(self.weight[0, 0, :, :])
        x, windows = self.cache

        padding = self.kernel_size - 1 if self.padding == 0 else self.kernel_size - 1-self.padding
        print("Dout: ", dout)
        dout_windows = getWindows(dout, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride, back=True)
        print("DOUT WINDOW: ",dout_windows)
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))
        print("Kernel: ", rot_kern)
        db = np.sum(dout, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        print("DX: ", dx)
        return db, dw, dx


def compare_conv2d(channel_in, channel_out, kernel_size, stride, padding):
    h, w = 2, 2
    X = np.random.rand(1, channel_in, h, w)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    h_out = (h + 2 * padding - kernel_size) // stride + 1
    w_out = (w + 2 * padding - kernel_size) // stride + 1
    y = np.random.rand(1, channel_out, h_out, w_out)

    # get the layers
    conv = Conv2D(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    conv_torch = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    conv_torch.weight = nn.Parameter(torch.from_numpy(conv.state_dict["kernels"].copy()))
    conv_torch.bias = nn.Parameter(torch.from_numpy(conv.state_dict["bias"].copy()))

    conv2 = Conv2D2(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=stride, padding=padding)
    conv2.weight = conv.state_dict["kernels"].copy()
    conv2.bias = conv.state_dict["bias"].copy()

    # forward
    y_pred = conv.forward(X)
    y_pred2 = conv2.forward(X)
    y_pred_torch = conv_torch(X_torch)
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())

    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    
    # backward
    dL_dy = y_pred - y.copy()
    dL_dX = conv.backward(dL_dy, lr=0.01)
    dL_dW = conv.grads["kernels"]
    dL_db = conv.grads["bias"]
    
    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)
    dL_dX_torch = X_torch.grad
    dL_dW_torch = conv_torch.weight.grad
    dL_db_torch = conv_torch.bias.grad

    dL_dy_2 = y_pred2 - y
    dL_dX_2 = conv2.backward(dL_dy_2)[2]

    print(dL_dX_2[0, 0, 0, :], dL_dX[0, 0, 0, :], dL_dX_torch.detach().numpy())
    # print(dL_dy_2[0, 0, 0, :], dL_dy[0, 0, 0, :], dL_dy_torch[0, 0, 0, :].detach().numpy())
    
    # print("Bias: ", conv2.bias, conv_torch.bias)
    # assert np.allclose(dL_dX_2, dL_dX), "2 -> Backward data don't match"

    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    assert dL_dW.shape == dL_dW_torch.shape, "Backward dW Shapes don't match"
    assert dL_db.shape == dL_db_torch.shape, "Backward db Shapes don't match"

    # print(dL_dX[1, 0, 0, 0], dL_dX_torch[1, 0, 0, 0].detach().numpy())
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())
    dw_compare = np.allclose(dL_dW, dL_dW_torch.detach().numpy())
    db_compare = np.allclose(dL_db, dL_db_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dw_compare, "Backward dW check failed"
    assert db_compare, "Backward db check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare and dw_compare and db_compare


# def test_conv1():
#     out = compare_conv2d(channel_in=3, channel_out=6, kernel_size=3, stride=1, padding=0) == True
#     assert out, "Basic Conv k=2, s=1, p=0 failed"
    
# def test_conv2():
#     out = compare_conv2d(channel_in=3, channel_out=6, kernel_size=3, stride=1, padding=2) == True
#     assert out, "Basic Conv k=3, s=1, p=2 failed"

# def test_conv3():
#     out = compare_conv2d(channel_in=1, channel_out=1, kernel_size=3, stride=2, padding=2) == True
#     assert out, "Basic Conv k=3, s=2, p=2 failed"

# if __name__ == "__main__":
#     # testing the conv layer with torch equivalent
#     import torch
#     import torch.nn as nn

#     X = np.random.randn(1, 6, 6, 6)
#     X_torch = torch.from_numpy(X); X_torch.requires_grad = True
#     y = np.random.randn(1, 4, 5, 5)

#     conv = Conv2D(in_channels=6, out_channels=4, kernel_size=2, stride=1, padding=0, bias=True)
#     conv_torch = nn.Conv2d(in_channels=6, out_channels=4, kernel_size=2, stride=1, padding=0, bias=True)
#     conv_torch.weight = nn.Parameter(torch.from_numpy(conv.state_dict["kernels"]))
#     conv_torch.bias = nn.Parameter(torch.from_numpy(conv.state_dict["bias"]))

#     # forward
#     y_pred = conv.forward(X)
#     y_pred_torch = conv_torch(X_torch)
    
#     print("CHECKING FORWARD:", np.allclose(y_pred, y_pred_torch.detach().numpy()))

#     # backward
#     dL_dy = y_pred - y
#     dL_dX = conv.backward(dL_dy, lr=0.01)
#     dL_dW = conv.grads["kernels"]
#     dL_db = conv.grads["bias"]
    
#     dL_dy_torch = y_pred_torch - torch.from_numpy(y)
#     y_pred_torch.backward(dL_dy_torch)
#     dL_dX_torch = X_torch.grad
#     dL_dW_torch = conv_torch.weight.grad
#     dL_db_torch = conv_torch.bias.grad
#     print(dL_dX.shape, dL_dX_torch.shape)
#     # print(dL_dW, dL_dW_torch)
#     # print(dL_db, dL_db_torch)

#     print("CHECKING BACKWARD X:", np.allclose(dL_dX, dL_dX_torch.detach().numpy()))
#     print("CHECKING BACKWARD W:", np.allclose(dL_dW, dL_dW_torch.detach().numpy()))
#     print("CHECKING BACKWARD B:", np.allclose(dL_db, dL_db_torch.detach().numpy()))


    
