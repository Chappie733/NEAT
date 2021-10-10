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
def softplus(x):
    return np.log(1+np.exp(-x))

@activation_logger
def sinh(x):
    return np.math.sinh(x)

@activation_logger
def cosh(x):
    return np.math.cosh(x)

@activation_logger
def bentidentity(x):
	return x+(np.sqrt(x**2+1)-1)/2

@activation_logger
def arctan(x):
    return np.math.atan(x)

@activation_logger
def bipolarsigmoid(x):
	return (1-np.exp(-x))/(1+np.exp(x))

@activation_logger
def sinc(x):
    return np.sinc(x)

@activation_logger
def softsign(x):
    return x/(1+np.abs(x))

@activation_logger
def relu(x):
    return np.maximum(x, 0)

@activation_logger
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))