import numpy as np

from model.Base import Base

class Linear(Base):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.name = "Linear"
        self.params = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
        }
        self.state_dict = self.initialize_parameters()

    def initialize_parameters(self):
        # kalming initialization
        # https://paperswithcode.com/method/he-initialization
        std = np.sqrt(2 / self.params["in_features"])
        weights = np.random.randn(self.params["out_features"], self.params["in_features"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_features"])
            return {"weights": weights, "bias": bias}
        return {"weights": weights}

    def forward(self, X):
        '''
        X shape should be (N, in_features)
        W shape is (out_features, in_features)
        so the output shape is (N, out_features)
        '''
        self.X = X
        output = np.dot(X, self.state_dict["weights"].T)
        if self.params["bias"]: output += self.state_dict["bias"]
        return output

    def backward(self, dL_dy, lr):
        '''
        dL_dy = gradient of the cost with respect to the output of the linear layer -> (bs, out_features)
        '''
        bs, _ = dL_dy.shape
        # gradient of the cost with respect to the weights
        dL_dW = np.dot(dL_dy.T, self.X) / bs  # (out_features, bs) * (bs, in_features) -> (out_features, in_features)
        # gradient of the cost with respect to the input
        dL_dX = np.dot(dL_dy, self.state_dict["weights"]) # (bs, out_features) * (out_features, in_features) -> (bs, in_features)
        # gradient of the cost with respect to the bias
        if self.params["bias"]: dL_db = np.sum(dL_dy, axis=0) / bs

        # update weights and bias
        self.grads = {"weights": dL_dW} 
        if self.params["bias"]: self.grads["bias"] = dL_db
        self.update_weights(lr)
        
        return dL_dX