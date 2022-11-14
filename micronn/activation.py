import numpy as np


class Activation:

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def get(activation):
        return activation


class Sigmoid(Activation):
    
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z):
        A = self.forward(Z)
        return A * (1 - A)


class Relu(Activation):

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z):
        return np.where(Z < 0, 0, 1)


class Leaky(Activation):

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, Z):
        return np.maximum(self.alpha * Z, Z)

    def backward(self, Z):
        return np.where(self.alpha < 0, self.alpha, 1)


class Tanh(Activation):

    def forward(self, Z):
        return np.divide(np.exp(Z) - np.exp(-Z), np.exp(Z) + np.exp(-Z))

    def backward(self, Z):
        A = self.forward(Z)
        return 1 - np.square(A)


class Softmax(Activation):

    def forward(self, Z):
        pass

    def backward(self, Z):
        pass
        