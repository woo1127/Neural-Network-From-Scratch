from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, axes):
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    
    xx, yy = np.meshgrid(x1grid, x2grid)
    
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
    grid = np.hstack((r1, r2))
    yhat = model(grid)
    yhat = np.where(yhat < 0.5, 0, 1)
    zz = yhat.reshape(xx.shape)
    
    axes.contourf(xx, yy, zz)
    axes.scatter(X[:, 0], X[:, 1], c=y)


def create_mini_batches(X, y, batch_size):
    X, y = X.T, y.T

    m = X.shape[0]
    num_of_batch = m // batch_size

    permutation = list(np.random.permutation(m))
    shuffle_X = X[permutation, :]
    shuffle_y = y[permutation, :]

    for t in range(num_of_batch):
        mini_x = shuffle_X[t * batch_size: (t + 1) * batch_size, :]
        mini_y = shuffle_y[t * batch_size: (t + 1) * batch_size, :]
        yield (mini_x.T, mini_y.T)
    
    if m % batch_size != 0:
        mini_x = shuffle_X[m // batch_size * batch_size: , :]
        mini_y = shuffle_y[m // batch_size * batch_size: , :]
        yield (mini_x.T, mini_y.T)


def iter_to_epochs_loss(loss_history, num_of_epochs):
    loss_batch_size = len(loss_history) // num_of_epochs
    loss_batch = []

    for i in range(num_of_epochs):
        mini_loss = loss_history[i * loss_batch_size: (i + 1) * loss_batch_size]
        loss_batch.append(np.sum(mini_loss) / loss_batch_size)

    return loss_batch


def reshape_dimension(X, y):
    if y.ndim == 1:
        return X.T, y.reshape((1, -1))
    return X.T, y.T
