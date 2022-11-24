import numpy as np
from micronn.utils import create_mini_batches, iter_to_epochs_loss, reshape_dimension
from tqdm import trange


class Model:

    def __init__(self, net, loss, optimizer, regularizer=None):
        self.net = net
        self.loss = loss
        self.loss_history = []
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.is_training = True

    def _forward(self, X):
        A = X
        for layer in self.net:
            if not self.is_training:
                layer.is_training = False
            A = layer.forward(A)
        return A

    def _backward(self, loss_grad):
        dA = loss_grad
        for layer in reversed(self.net):
            if not self.is_training:
                layer.is_training = False
            dA = layer.backward(dA, self.m)

    def _compute_loss(self, pred, target):
        loss_grad = self.loss.grad(pred, target)
        loss = self.loss.loss(pred, target, self.m)
        loss = np.squeeze(loss)

        self.loss_history.append(loss)
        return loss_grad

    def _update(self):
        self.optimizer.update(self.net)

    def fit(self, X, y, batch_size=32, epochs=10):
        X, y = reshape_dimension(X, y)
        
        for _ in trange(epochs):
            for mini_x, mini_y in create_mini_batches(X, y, batch_size):
                self.m = mini_x.shape[1]
                pred = self._forward(mini_x)
                loss_grad = self._compute_loss(pred, mini_y)
                self._backward(loss_grad)
                self._update()
        self.loss_history = iter_to_epochs_loss(self.loss_history, epochs)

    def predict(self, X_test):
        self.is_training = False

        X_test = X_test.T
        pred = self._forward(X_test)

        return pred
        
