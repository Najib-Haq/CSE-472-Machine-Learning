import numpy as np
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.Conv2d import Conv2D

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


    
