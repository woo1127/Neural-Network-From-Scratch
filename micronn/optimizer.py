import numpy as np


class Optimizer:

    def __init__(self):
        self.is_init = False

    def update(self):
        raise NotImplementedError

    def init_optim_grad(self, dW, db):
        return np.where(dW, 0, dW), np.where(db, 0, db)

    
class GD(Optimizer):

    def __init__(self, lr=0.01, momentum=0):
        super().__init__()
        self.lr = lr
        self.beta = momentum
        self.v = {}

    def update(self, net):
        for layer, i in zip(net, range(1, len(net))):
            W, b = layer.get_weights()
            dW, db = layer.get_weights_grad()

            if np.any(W) and np.any(dW):
                if not self.is_init:
                    self.v['dW' + str(i)], self.v['db' + str(i)] = self.init_optim_grad(dW, db)

                self.v['dW' + str(i)] = self.beta * self.v['dW' + str(i)] + (1 - self.beta) * dW
                self.v['db' + str(i)] = self.beta * self.v['db' + str(i)] + (1 - self.beta) * db

                W -= self.lr * self.v['dW' + str(i)]
                b -= self.lr * self.v['db' + str(i)]

                layer.set_weights((W, b))

        self.is_init = True


class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta = rho
        self.epsilon = epsilon
        self.s = {}

    def update(self, net):

        for layer, i in zip(net, range(len(net))):
            W, b = layer.get_weights()
            dW, db = layer.get_weights_grad()

            if np.any(W) and np.any(dW):
                if not self.is_init:
                    self.s['dW' + str(i)], self.s['db' + str(i)] = self.init_optim_grad(dW, db)

                self.s['dW' + str(i)] = self.beta * self.s['dW' + str(i)] + (1 - self.beta) * np.square(dW)
                self.s['db' + str(i)] = self.beta * self.s['db' + str(i)] + (1 - self.beta) * np.square(db)

                W -= self.lr * dW / np.sqrt(self.s['dW' + str(i)] + self.epsilon)
                b -= self.lr * db / np.sqrt(self.s['db' + str(i)] + self.epsilon)

                layer.set_weights((W, b))
                
        self.is_init = True

    
class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.99, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = {}
        self.s = {}
        self.iter_t = 0
    
    def update(self, net):
        self.iter_t += 1

        for layer, i in zip(net, range(len(net))):
            W, b = layer.get_weights()
            dW, db = layer.get_weights_grad()

            if np.any(W) and np.any(dW):
                if not self.is_init:
                    self.v['dW' + str(i)], self.v['db' + str(i)] = self.init_optim_grad(dW, db)
                    self.s['dW' + str(i)], self.s['db' + str(i)] = self.init_optim_grad(dW, db)

                bias_v = (1 - self.beta1 ** self.iter_t)
                bias_s = (1 - self.beta2 ** self.iter_t)

                self.v['dW' + str(i)] = self.beta1 * self.v['dW' + str(i)] + (1 - self.beta1) * dW / bias_v
                self.v['db' + str(i)] = self.beta1 * self.v['db' + str(i)] + (1 - self.beta1) * db / bias_v
                self.s['dW' + str(i)] = self.beta2 * self.s['dW' + str(i)] + (1 - self.beta2) * np.square(dW) / bias_s
                self.s['db' + str(i)] = self.beta2 * self.s['db' + str(i)] + (1 - self.beta2) * np.square(db) / bias_s

                W -= self.lr * self.v['dW' + str(i)] / np.sqrt(self.s['dW' + str(i)] + self.epsilon)
                b -= self.lr * self.v['db' + str(i)] / np.sqrt(self.s['db' + str(i)] + self.epsilon)

                layer.set_weights((W, b))
                
        self.is_init = True
