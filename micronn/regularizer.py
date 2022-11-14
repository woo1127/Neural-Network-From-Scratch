import numpy as np


class Regularizer:

    def loss(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class L1(Regularizer):

    def loss(self):
        pass

    def grad(self):
        pass


class L2(Regularizer):

    def __init__(self, lambd):
        self.lambd = lambd

    def loss(self, param, m):
        self.all_W = [param[i] for i in list(param.keys()) if "W" in i]
        return self.lambd / (2 * m) * np.sum( [np.sum(np.square(w)) for w in self.all_W] )

    def grad(self, w):
        return self.lambd / 2 * w 