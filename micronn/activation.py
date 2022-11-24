import numpy as np


class Activation:

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def get(activation):
        return activation


class Sigmoid(Activation):
    
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, dA, Z):
        A = self.forward(Z)
        return dA * A * (1 - A)

class Relu(Activation):

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, dA, Z):
        return dA * np.where(Z < 0, 0, 1)


class Leaky(Activation):

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, Z):
        return np.maximum(self.alpha * Z, Z)

    def backward(self, dA, Z):
        return dA * np.where(Z < 0, self.alpha * Z, 1)


class Tanh(Activation):

    def forward(self, Z):
        return np.divide(np.exp(Z) - np.exp(-Z), np.exp(Z) + np.exp(-Z))

    def backward(self, dA, Z):
        A = self.forward(Z)
        return dA * (1 - np.square(A))


class Softmax(Activation):

    def forward(self, Z):
        Z_exp = np.exp(Z - np.max(Z))
        return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)
        
    def backward(self, dA, Z):
        # initialise a random array to store the value of gradient
        dL_dZ = np.zeros_like(Z, dtype=np.float32)

        # transpose the values to loop the column instead of row
        for dl_da, z, i in zip(dA.T, Z.T, range(Z.shape[1])):
            # transform from 1D to 2D
            z = np.expand_dims(z, 1)
            dl_da = np.expand_dims(dl_da, 1)

            s = self.forward(z)
            da_dz = np.diagflat(s) - np.dot(s, s.T)
            dl_dz = - np.dot(da_dz, dl_da)
            # flatten from 2D to 1D and store the value of gradient
            dL_dZ[:, i] = dl_dz.flatten()
            
        return dL_dZ
        
        
        
        