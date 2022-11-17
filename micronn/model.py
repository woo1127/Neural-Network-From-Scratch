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

    def _forward(self, X):
        A = X
        for l in range(1, len(self.net)):
            A = self.net[l].forward(A)
        return A

    def _backward(self, loss_grad):
        dA = loss_grad
        for l in range(len(self.net) - 1, 0, -1):
            dA = self.net[l].backward(dA, self.m)

    def _compute_loss(self, pred, target):
        loss_grad = self.loss.grad(pred, target)
        loss = self.loss.loss(pred, target, self.m)
        loss = np.squeeze(loss)

        self.loss_history.append(loss)
        return loss_grad

    def _update(self):
        self.optimizer.update(self.net)

    def fit(self, X, y, batch_size=32, epochs=10):
        X, y = X.T, y.reshape((1, -1))
        self.net.insert(0, Input([X.shape[0], X.shape[1]]))
        
        for _ in trange(epochs):
            for mini_x, mini_y in create_mini_batches(X, y, batch_size):
                self.m = mini_x.shape[1]
                pred = self._forward(mini_x)
                loss_grad = self._compute_loss(pred, mini_y)
                self._backward(loss_grad, mini_y)
                self._update()
        self.loss_history = iter_to_epochs_loss(self.loss_history, epochs)

    def predict(self, X_test):
        X_test = X_test.T
        pred = self._forward(X_test)

        return pred.ravel()
        
