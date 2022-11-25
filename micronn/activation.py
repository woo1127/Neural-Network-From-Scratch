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
        exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)
        
    def backward(self, dA, Z):
        S = self.forward(Z)
        zeros = np.eye(S.shape[0])

        diagflat = np.einsum('ij,jk->kij', zeros, S)
        ss = np.einsum('ij,jk->jik', S, S.T)

        dA_dZ = diagflat - ss
        dA = np.expand_dims(dA, 1).T

        dZ = - np.matmul(dA, dA_dZ).T
        dZ = dZ.reshape(dZ.shape[0], dZ.shape[1] * dZ.shape[2])

        return dZ
        
    # def backward(self, dA, Z):
    #     # initialise a random array to store the value of gradient
    #     dZ = np.zeros_like(Z, dtype=np.float32)

    #     # transpose the values to loop the column instead of row
    #     for da, z, i in zip(dA.T, Z.T, range(Z.shape[1])):
    #         # transform from 1D to 2D
    #         z = np.expand_dims(z, 1)
    #         da = np.expand_dims(da, 1)

    #         s = self.forward(z)
    #         da_dz = np.diagflat(s) - np.dot(s, s.T)
    #         dz = - np.dot(da_dz, da)
    #         # flatten from 2D to 1D and store the value of gradient
    #         dZ[:, i] = dz.flatten()
        
    #     return dZ

        
    