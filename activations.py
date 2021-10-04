import numpy as np

ACTIVATIONS = {}

def activation_logger(activation):
    ACTIVATIONS[activation.__name__] = activation
    return activation

@activation_logger
def none(x):
    return x

@activation_logger
def tanh(x):
    return np.tanh(x)

@activation_logger
def sigmoid(x):
    return 1/(1+np.exp(-x))

@activation_logger
def ReLu(x):
    return np.maximum(x, 0)