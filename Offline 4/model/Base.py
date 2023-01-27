import numpy as np

from utils import *

class Base:
    def __init__(self):
        self.params = {}
        self.state_dict = {}
        self.grads = {}
        self.name = "Base"

    def __str__(self):
        return self.name + "(" + str(self.params) + ")"

    def forward(self, X):
        pass

    def __call__(self, X):
        return self.forward(X)

    def update_weights(self, lr):
        for key in self.state_dict:
            self.state_dict[key] -= lr * self.grads[key]
        # print(self.name + " : " , self.state_dict)

class Derived(Base):
    def __init__(self):
        super().__init__()
        self.name = "Derived"
        self.params = {
            "kernel_size": 3,
            "stride": 1,
        }

    def forward(self, X):
        print("Forwarding...", X)

if __name__ == "__main__":
    a = Derived()
    print(a)
    a("hey")