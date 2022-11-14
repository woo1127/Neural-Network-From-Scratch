import numpy as np


class Initializer:

    def init(self):
        raise NotImplementedError

    @staticmethod
    def get(initializer):
        return initializer.init


class HeNormal(Initializer):

    def init(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[1])


class XiavierNormal(Initializer):
    
    def init(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0] + shape[1])


class Zeros(Initializer):

    def init(self, shape):
        return np.zeros((shape[0], shape[1]))


class Ones(Initializer):

    def init(self, shape):
        return np.ones((shape[0], shape[1]))