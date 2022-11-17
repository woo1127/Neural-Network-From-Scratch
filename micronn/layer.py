import numpy as np
from micronn.initializer import Initializer, HeNormal, Zeros
from micronn.activation import Activation, Relu


class Layer:

    def __init__(self):
        self.is_init = False
        self.weights = {}
        self.params = {}
        self.grads = {}

    def get_weights(self):
        if self.is_init:
            return self.weights
        pass

    def get_weights_grad(self):
        if self.is_init:
            return {i: self.grads[i] for i in self.grads.keys() if 'dW' in i or 'db' in i}
        pass

    def set_weights(self, weights):
        self.weights = weights

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
        super().__init__()
        self.units = units 
        self.activation = Activation.get(activation)
        self.w_init = w_init
        self.b_init = b_init

    def init_weights(self, curr, prev):
        self.weights['W'] = Initializer.get(self.w_init)([curr, prev])
        self.weights['b'] = Initializer.get(self.b_init)([curr, 1])
        
    def forward(self, A):
        if not self.is_init:
            self.init_weights(self.units, A.shape[0])
            self.is_init = True

        self.params['A_prev'] = A
        self.params['Z'] = np.dot(self.weights['W'], A) + self.weights['b']

        A = self.activation.forward(self.params['Z'])
        return A

    def backward(self, dA, m):
        self.grads['dZ'] = dA * self.activation.backward(self.params['Z'])
        self.grads['dW'] = np.dot(self.grads['dZ'], self.params['A_prev'].T) / m
        self.grads['db'] = np.sum(self.grads['dZ'], axis=1, keepdims=True) / m

        dA = np.dot(self.weights['W'].T, self.grads['dZ'])
        return dA











        

