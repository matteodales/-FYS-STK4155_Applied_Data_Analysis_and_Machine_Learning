
# implements the layer of the neural network
import numpy as np

class Layer():

    """ Implements a layer for a Feed-Forward Neural Network """

    def __init__(self, size, act_fun, act_fun_grad, initial_bias = 0.001):

        self._size = size
        self._act_fn = act_fun
        self._act_fn_grad = act_fun_grad
        self._weights = None
        self._initial_bias = initial_bias
        self._biases = np.ones((self._size, 1)) * initial_bias


    def create_weights(self, input_size, rng):
        
        # initializes the weights for the layers between 0 and 1
        self._weights = rng.uniform(-1,1,(self._size, input_size))

    def forward(self, inputs):
        
        # returns the output and activated output of the layer with the current weights and biases
        z = (self._weights @ inputs.T + self._biases).T
        return self._act_fn(z), z

    def backward(self, activated_inputs, inputs, error, prev_act_fn_grad, learning_rate, regularization):

        # Compute gradients
        # Simple gradient descent
        weights_gradient = (activated_inputs.T @ error) + regularization * self._weights.T
        bias_gradient = error + regularization * self._biases.T
        
        # Adjust weights and biases
        self._weights -= learning_rate * weights_gradient.T
        self._biases -= learning_rate * bias_gradient.T

        # Return the estimated error in inputs
        return np.multiply((error @ self._weights), prev_act_fn_grad(inputs))