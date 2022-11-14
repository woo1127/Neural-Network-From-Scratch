import numpy as np
from micronn.initializer import Initializer, HeNormal, Zeros
from micronn.activation import Activation, Relu


class Layer:

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Input(Layer):
    
    def __init__(self, shape):
        self.shape = shape
        self.units = self.shape[0]


class Dense(Layer):

    def __init__(self, units, activation=Relu(), w_init=HeNormal(), b_init=Zeros()):
        self.units = units 
        self.activation = Activation.get(activation)
        self.w_init = w_init
        self.b_init = b_init

    def init_param(self, curr, prev):
        w = Initializer.get(self.w_init)([curr, prev])
        b = Initializer.get(self.b_init)([curr, 1])
        return w, b
        
    def forward(self, w, b, A):
        Z = np.dot(w, A) + b
        A = self.activation.forward(Z)
        return Z, A

    def backward(self, dA, A, Z, w, m):
        dZ = dA * self.activation.backward(Z)
        dW = np.dot(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(w.T, dZ)
        return dZ, dW, db, dA











        

