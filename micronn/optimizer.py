import numpy as np


class Optimizer:

    def __init__(self):
        self.is_init = False

    def update(self):
        raise NotImplementedError

    def init_optim_grad(self, grads):
        return np.where(grads['dW'], 0, grads['dW']), np.where(grads['db'], 0, grads['db'])

    
class GD(Optimizer):

    def __init__(self, lr=0.01, momentum=0):
        super().__init__()
        self.lr = lr
        self.beta = momentum
        self.v = {}

    def update(self, net):
        for layer, i in zip(net, range(1, len(net))):
            weights = layer.get_weights()
            grads = layer.get_weights_grad()

            if grads and weights:
                if not self.is_init:
                    self.v['dW' + str(i)], self.v['db' + str(i)] = self.init_optim_grad(grads)

                self.v['dW' + str(i)] = self.beta * self.v['dW' + str(i)] + (1 - self.beta) * grads['dW']
                self.v['db' + str(i)] = self.beta * self.v['db' + str(i)] + (1 - self.beta) * grads['db']

                weights['W'] -= self.lr * self.v['dW' + str(i)]
                weights['b'] -= self.lr * self.v['db' + str(i)]

                layer.set_weights({'W': weights['W'], 'b': weights['b']})

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
            weights = layer.get_weights()
            grads = layer.get_weights_grad()

            if grads and weights:
                if not self.is_init:
                    self.s['dW' + str(i)], self.s['db' + str(i)] = self.init_optim_grad(grads)

                self.s['dW' + str(i)] = self.beta * self.s['dW' + str(i)] + (1 - self.beta) * np.square(grads['dW'])
                self.s['db' + str(i)] = self.beta * self.s['db' + str(i)] + (1 - self.beta) * np.square(grads['db'])

                weights['W'] -= self.lr * grads['dW'] / np.sqrt(self.s['dW' + str(i)] + self.epsilon)
                weights['b'] -= self.lr * grads['db'] / np.sqrt(self.s['db' + str(i)] + self.epsilon)

                layer.set_weights({'W': weights['W'], 'b': weights['b']})
                
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
            weights = layer.get_weights()
            grads = layer.get_weights_grad()

            if grads and weights:
                if not self.is_init:
                    self.v['dW' + str(i)], self.v['db' + str(i)] = self.init_optim_grad(grads)
                    self.s['dW' + str(i)], self.s['db' + str(i)] = self.init_optim_grad(grads)

                bias_corrected_v = (1 - self.beta1 ** self.iter_t)
                bias_corrected_s = (1 - self.beta2 ** self.iter_t)

                self.v['dW' + str(i)] = self.beta1 * self.v['dW' + str(i)] + (1 - self.beta1) * grads['dW'] / bias_corrected_v
                self.v['db' + str(i)] = self.beta1 * self.v['db' + str(i)] + (1 - self.beta1) * grads['db'] / bias_corrected_v
                self.s['dW' + str(i)] = self.beta2 * self.s['dW' + str(i)] + (1 - self.beta2) * np.square(grads['dW']) / bias_corrected_s
                self.s['db' + str(i)] = self.beta2 * self.s['db' + str(i)] + (1 - self.beta2) * np.square(grads['db']) / bias_corrected_s

                weights['W'] -= self.lr * self.v['dW' + str(i)] / np.sqrt(self.s['dW' + str(i)] + self.epsilon)
                weights['b'] -= self.lr * self.v['db' + str(i)] / np.sqrt(self.s['db' + str(i)] + self.epsilon)

                layer.set_weights({'W': weights['W'], 'b': weights['b']})
                
        self.is_init = True
