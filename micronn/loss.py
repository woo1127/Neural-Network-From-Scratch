import numpy as np


class Loss:

    def cost(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class BinaryCrossentropy(Loss):

    def loss(self, pred, target, m):
        return - np.dot(target, np.log(pred + 1e-8).T) - np.dot(1 - target, np.log(1 - target + 1e-8).T) * 1 / m

    def grad(self, pred, target):
        return - (np.divide(target, pred + 1e-8) - np.divide(1 - target, 1 - pred + 1e-8))

class CategoricalCrossentropy(Loss):

    def loss(self, pred, target, m):
        return np.sum(-np.sum(target * np.log(pred + 1e-8), axis=0, keepdims=True)) / m

    def grad(self, pred, target):
        return target / pred

