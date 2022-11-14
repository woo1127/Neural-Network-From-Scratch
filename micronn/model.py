import numpy as np
from micronn.layer import Input
from micronn.utils import create_mini_batches, iter_to_epochs_loss
from tqdm import trange


class Model:

    def __init__(self, net, loss, optimizer, regularizer=None):
        self.net = net
        self.loss = loss
        self.loss_history = []
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.param = {}
        self.grad = {}

    def _init_param(self):
        for l in range(1, len(self.net)):
            prev_units = self.net[l - 1].units
            curr_units = self.net[l].units

            self.param['W' + str(l)], \
            self.param['b' + str(l)], \
            = self.net[l].init_param(curr_units, prev_units)

    def _forward(self, X):
        self.param['A0'] = X

        for l in range(1, len(self.net)):
            w = self.param['W' + str(l)]
            b = self.param['b' + str(l)]
            prev_A = self.param['A' + str(l - 1)]

            self.param['Z' + str(l)], \
            self.param['A' + str(l)], \
            = self.net[l].forward(w, b, prev_A)

    def _backward(self, y):
        last_L = str(len(self.net) - 1)
        pred = self.param['A' + last_L]
        self.grad['dA' + last_L] = self.loss.grad(pred, y)

        for l in range(len(self.net) - 1, 0, -1):
            w = self.param['W' + str(l)]
            z = self.param['Z' + str(l)]
            prev_A = self.param['A' + str(l - 1)]
            dA = self.grad['dA' + str(l)]

            self.grad['dZ' + str(l)], \
            self.grad['dW' + str(l)], \
            self.grad['db' + str(l)], \
            self.grad['dA' + str(l - 1)], \
            = self.net[l].backward(dA, prev_A, z, w, self.m)

            if self.regularizer:
                self.grad['dW' + str(l)] += self.regularizer.grad(w)

    def _compute_loss(self, pred, target):
        loss = self.loss.loss(pred, target, self.m)
        loss = np.squeeze(loss)

        if self.regularizer:
            loss += self.regularizer.loss(self.param, self.m)
        self.loss_history.append(loss)

    def _update(self):
        self.param = self.optimizer.update(self.param, self.grad, len(self.net))

    def fit(self, X, y, batch_size=32, epochs=10):
        X, y = X.T, y.reshape((1, -1))
        self.net.insert(0, Input([X.shape[0], X.shape[1]]))
        self._init_param()
        
        for _ in trange(epochs):
            for mini_x, mini_y in create_mini_batches(X, y, batch_size):
                self.m = mini_x.shape[1]
                self._forward(mini_x)
                self._backward(mini_y)
                self._compute_loss(self.param['A' + str(len(self.net) - 1)], mini_y)
                self._update()
        self.loss_history = iter_to_epochs_loss(self.loss_history, epochs)

    def predict(self, X_test):
        X_test = X_test.T
        self.param["A0"] = X_test
        self._forward(X_test)

        self.y_pred = self.param["A" + str(len(self.net) - 1)]
        self.y_pred = np.where(self.y_pred >= 0.5, 1, 0)

        return self.y_pred.ravel()

