import numpy as np


class Loss:

    def cost(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class BinaryCrossentropy(Loss):

    def loss(self, pred, target, m, epsilon=1e-8):
        return - np.dot(target, np.log(pred + epsilon).T) - np.dot(1 - target, np.log(1 - target + epsilon).T) * 1 / m

    def grad(self, pred, target):
        return - (np.divide(target, pred) - np.divide(1 - target, 1 - pred))


