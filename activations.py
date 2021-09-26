import numpy as np

def none(x):
    return x

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLu(x):
    return np.maximum(x, 0)