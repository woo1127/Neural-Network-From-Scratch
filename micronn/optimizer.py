import numpy as np


class Optimizer:

    def update(self):
        raise NotImplementedError

    def init_optim_grad(self, grad):
        self.batch_t = 0
        self.s = {}
        self.v = {}

        for i in list(grad.keys()):
            if 'dW' in i or 'db' in i:
                self.s[i] = np.zeros((grad[i].shape[0], grad[i].shape[1]))
                self.v[i] = np.zeros((grad[i].shape[0], grad[i].shape[1]))


class GD(Optimizer):

    def __init__(self, lr=0.01, momentum=0):
        self.lr = lr
        self.beta = momentum
        
    def update(self, param, grad, L):
        for l in range(1, L):
            self.v['dW' + str(l)] = self.beta * self.v['dW' + str(l)] + (1 - self.beta) * grad['dW' + str(l)]
            self.v['db' + str(l)] = self.beta * self.v['db' + str(l)] + (1 - self.beta) * grad['db' + str(l)]

            param['W' + str(l)] -= self.lr * self.v['dW' + str(l)]
            param['b' + str(l)] -= self.lr * self.v['db' + str(l)]

        return param


class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-7):
        self.lr = lr
        self.beta = rho
        self.epsilon = epsilon

    def update(self, param, grad, L):
        for l in range(1, L):
            self.s['dW' + str(l)] = self.beta * self.s['dW' + str(l)] + (1 - self.beta) * np.square(grad['dW' + str(l)])
            self.s['db' + str(l)] = self.beta * self.s['db' + str(l)] + (1 - self.beta) * np.square(grad['db' + str(l)])

            param['W' + str(l)] -= self.lr * grad['dW' + str(l)] / np.sqrt(self.s['dW' + str(l)] + self.epsilon)
            param['b' + str(l)] -= self.lr * grad['db' + str(l)] / np.sqrt(self.s['db' + str(l)] + self.epsilon)

        return param


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.99, epsilon=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, param, grad, L):
        self.batch_t += 1

        for l in range(1, L):
            bias_corrected_v = (1 - self.beta1 ** self.batch_t)
            bias_corrected_s = (1 - self.beta2 ** self.batch_t)

            self.v['dW' + str(l)] = self.beta1 * self.v['dW' + str(l)] + (1 - self.beta1) * grad['dW' + str(l)] / bias_corrected_v
            self.v['db' + str(l)] = self.beta1 * self.v['db' + str(l)] + (1 - self.beta1) * grad['db' + str(l)] / bias_corrected_v
            self.s['dW' + str(l)] = self.beta2 * self.s['dW' + str(l)] + (1 - self.beta2) * np.square(grad['dW' + str(l)]) / bias_corrected_s
            self.s['db' + str(l)] = self.beta2 * self.s['db' + str(l)] + (1 - self.beta2) * np.square(grad['db' + str(l)]) / bias_corrected_s

            param['W' + str(l)] -= self.lr * self.v['dW' + str(l)] / np.sqrt(self.s['dW' + str(l)] + self.epsilon)
            param['b' + str(l)] -= self.lr * self.v['db' + str(l)] / np.sqrt(self.s['db' + str(l)] + self.epsilon)

        return param
