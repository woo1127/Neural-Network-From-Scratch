import numpy as np
from micronn.initializer import Initializer, HeNormal, Zeros
from micronn.activation import Activation, Relu


class Layer:

    def __init__(self):
        self.is_init = False
        self.is_training = True

    def get_weights(self):
        if self.is_init:
            return self.W, self.b
        return None, None

    def set_weights(self, weights):
        self.W, self.b = weights

    def get_weights_grad(self):
        if self.is_init:
            return self.dW, self.db
        return None, None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, units, activation=Relu(), w_init=HeNormal(), b_init=Zeros()):
        super().__init__()
        self.units = units 
        self.activation = Activation.get(activation)
        self.w_init = w_init
        self.b_init = b_init

    def init_weights(self, curr, prev):
        self.W = Initializer.get(self.w_init)([curr, prev])
        self.b = Initializer.get(self.b_init)([curr, 1])
        
    def forward(self, A):
        if not self.is_init:
            self.init_weights(self.units, A.shape[0])
            self.is_init = True

        self.A_prev = A
        self.Z = np.dot(self.W, A) + self.b

        A = self.activation.forward(self.Z)
        return A

    def backward(self, dA, m):
        self.dZ = dA * self.activation.backward(self.Z)
        self.dW = np.dot(self.dZ, self.A_prev.T) / m
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / m

        dA = np.dot(self.W.T, self.dZ)

        return dA


class Dropout(Layer):

    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

        if not self.is_training:
            self.keep_prob = 1.0

    def forward(self, A):
        self.A_prev = A
        mask = (np.random.rand(*A.shape) < self.keep_prob).astype('int32')
        A = A * mask / self.keep_prob
        return A

    def backward(self, dA, m):
        mask = (np.random.rand(*dA.shape) < self.keep_prob).astype('int32')
        dA = dA * mask / self.keep_prob
        return dA

