import numpy as np


class Optimizer:

    def __init__(self):
        self.is_init = False
        self.params = {}
        self.grads = {}
        self.iter_t = 0
        self.s = {}
        self.v = {}

    def update(self):
        raise NotImplementedError

    def init_optim_grad(self, grads):
        for i in grads.keys():
            if 'dW' in i or 'db' in i:
                self.v[i] = np.where(grads[i], 0, grads[i])
                self.s[i] = np.where(grads[i], 0, grads[i])
    
    def load_grads(self, net):
        grads = {}

        for l in range(1, len(net)):
            layer_grads = net[l].get_weights_grad()

            if layer_grads:
                grads['dW' + str(l)] = layer_grads['dW']
                grads['db' + str(l)] = layer_grads['db']
        return grads


class GD(Optimizer):

    def __init__(self, lr=0.01, momentum=0):
        super().__init__()
        self.lr = lr
        self.beta = momentum

    def update(self, net):
        grads = self.load_grads(net)

        if not self.is_init:
            self.init_optim_grad(grads)
            self.is_init = True
                    
        for l in range(1, len(net)):
            self.v['dW' + str(l)] = self.beta * self.v['dW' + str(l)] + (1 - self.beta) * grads['dW' + str(l)]
            self.v['db' + str(l)] = self.beta * self.v['db' + str(l)] + (1 - self.beta) * grads['db' + str(l)]
            
            weights = net[l].get_weights()
            if weights:
                weights['W'] -= self.lr * self.v['dW' + str(l)]
                weights['b'] -= self.lr * self.v['db' + str(l)]

                net[l].set_weights({'W': weights['W'], 'b': weights['b']})


class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta = rho
        self.epsilon = epsilon

    def update(self, net):
        grads = self.load_grads(net)

        if not self.is_init:
            self.init_optim_grad(grads)
            self.is_init = True

        for l in range(1, len(net)):
            self.s['dW' + str(l)] = self.beta * self.s['dW' + str(l)] + (1 - self.beta) * np.square(grads['dW' + str(l)])
            self.s['db' + str(l)] = self.beta * self.s['db' + str(l)] + (1 - self.beta) * np.square(grads['db' + str(l)])

            weights = net[l].get_weights()
            if weights:
                weights['W'] -= self.lr * grads['dW' + str(l)] / np.sqrt(self.s['dW' + str(l)] + self.epsilon)
                weights['b'] -= self.lr * grads['db' + str(l)] / np.sqrt(self.s['db' + str(l)] + self.epsilon)

                net[l].set_weights({'W': weights['W'], 'b': weights['b']})

    
class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.99, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, net):
        grads = self.load_grads(net)

        if not self.is_init:
            self.init_optim_grad(grads)
            self.is_init = True

        self.iter_t += 1
        
        for l in range(1, len(net)):
            bias_corrected_v = (1 - self.beta1 ** self.iter_t)
            bias_corrected_s = (1 - self.beta2 ** self.iter_t)

            self.v['dW' + str(l)] = self.beta1 * self.v['dW' + str(l)] + (1 - self.beta1) * grads['dW' + str(l)] / bias_corrected_v
            self.v['db' + str(l)] = self.beta1 * self.v['db' + str(l)] + (1 - self.beta1) * grads['db' + str(l)] / bias_corrected_v
            self.s['dW' + str(l)] = self.beta2 * self.s['dW' + str(l)] + (1 - self.beta2) * np.square(grads['dW' + str(l)]) / bias_corrected_s
            self.s['db' + str(l)] = self.beta2 * self.s['db' + str(l)] + (1 - self.beta2) * np.square(grads['db' + str(l)]) / bias_corrected_s

            weights = net[l].get_weights()
            if weights:
                weights['W'] -= self.lr * self.v['dW' + str(l)] / np.sqrt(self.s['dW' + str(l)] + self.epsilon)
                weights['b'] -= self.lr * self.v['db' + str(l)] / np.sqrt(self.s['db' + str(l)] + self.epsilon)

                net[l].set_weights({'W': weights['W'], 'b': weights['b']})
            
