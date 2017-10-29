import numpy as np

def sigmoid(x):
    """Sigmoid transfer function."""
    return 1. / (1. + np.exp(-x))

def reLU(x):
    """Rectifier transfer function."""
    return x * (x > 0)

def leakyReLU(x):
    """Leaky rectifier transfer function."""
    return x * (x > 0) + (.1 * x) * (x < 0)

def tanh(x):
    """Hyperbollic tan transfer function."""
    return np.tanh(x)

