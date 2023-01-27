import numpy as np
import pickle

from model.Conv2d import *
from model.MaxPool2d import *
from model.Linear import *
from model.Activations import *

class Model:
    def __init__(self, config):
        self.config = config
        self.layers = self.create_model()
        # self.loss = nn.SoftmaxCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, dL_dy, lr):
        # print("INPUT BACKWARD: ",dL_dy.shape)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy, lr)
        return dL_dy

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        print_data = ""
        for i, layer in enumerate(self.layers):
            print_data += f"Layer {i}: " + str(layer) + "\n"
        return print_data

    def create_model(self):
        model = []

        for layer in self.config:
            name = layer[0]
            params = layer[1]

            if name == "Conv2D":
                model.append(Conv2D(**params))
            elif name == "MaxPool2D":
                model.append(MaxPool2D(**params))
            # elif name == "Flatten":
            #     model.append(nn.Flatten())
            elif name == "Linear":
                model.append(Linear(params[0], params[1]))
            # elif name == "ReLU":
            #     model.append(ReLU())
            elif name == "Softmax":
                model.append(Softmax())
        return model

    def save_model(self, path):
        params = []
        for layer in self.layers:
            params.append(layer.state_dict)
        with open(path, "wb") as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            layer.state_dict = params[i]
        print("Successfully loaded from " + path)
        