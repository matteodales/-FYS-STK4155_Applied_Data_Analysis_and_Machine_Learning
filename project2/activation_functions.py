# Activation functions

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(x):
    return np.multiply((x >= 0), x)

def ReLU_grad(x):
    return (x >= 0)*1

def leakyReLU(x):
    return np.multiply((x >= 0), x) + np.multiply((x < 0), x * 0.1)

def leakyReLU_grad(x):
    return (x >= 0)*1 + (x < 0)*0.1

def linear(x):
    return x

def linear_grad(x):
    return np.ones(np.shape(x))

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x)*np.tanh(x)