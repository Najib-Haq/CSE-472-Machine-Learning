import numpy as np
import pickle

from model.nn.Conv2d import *
from model.nn.MaxPool2d import *
from model.nn.Linear import *
from model.nn.Flatten import *
from model.nn.Activations import *

class Model:
    def __init__(self, config=False, model_layers=[]):
        self.config = config['model']
        if self.config: self.layers = self.create_model()
        else: self.layers = model_layers

        if config:
            print("Testing model shapes with random X: ")
            X = np.random.randn(1, 3, config['augment']['img_shape'][0], config['augment']['img_shape'][1])
            self.forward(X, debug=True)
            print('#'*50)

    def forward(self, X, debug=False):
        if debug: print("Input X: -> \t\t", X.shape)
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if debug: print(f"Layer {i}: {layer.name} ->\t", X.shape)
        return X

    def backward(self, dL_dy):
        # print("INPUT BACKWARD: ",dL_dy.shape)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        print_data = "MODEL LAYERS & PARAMETERS: \n"
        for i, layer in enumerate(self.layers):
            print_data += f"Layer {i}: " + str(layer) + "\n"
        return print_data

    def create_model(self):
        model = []

        for layer in self.config:
            name = layer[0]
            params = layer[1]

            if name == "Conv2D":
                model.append(Conv2D(*params))
            elif name == "MaxPool2D":
                model.append(MaxPool2D(*params))
            elif name == "Flatten":
                model.append(Flatten())
            elif name == "Linear":
                model.append(Linear(None, params[0], lazy_init=True))
            elif name == "ReLU":
                model.append(ReLU())
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

    # makes model trainable -> stores cache
    def train(self):
        for layer in self.layers:
            layer.trainable = True

    # makes model untrainable -> doesnt store cache
    def eval(self):
        for layer in self.layers:
            layer.trainable = False
        