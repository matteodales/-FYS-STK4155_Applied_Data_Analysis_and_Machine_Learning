
from Layer import Layer
import numpy as np

# creates the Multi Layer Perceptron model

class NeuralNetwork:

    """ Implements a Feed-Forward Neural Network """
    
    def __init__(self, input_size, cost_function_grad, random_state):

        # number of features in the input X matrix
        self._input_size = input_size

        # random number generator to make results reproducible
        self.generator = np.random.default_rng(seed=random_state)
        
        # gradient of the cost function considered
        self.cost_function_grad = cost_function_grad

        # list of layers (starts out empty)
        self.layers = list()
        
    
    def add_layer(self, layer):

        # adds a layer to the network

        if len(self.layers) > 0:
            n_inputs = self.layers[-1]._size
        else:
            n_inputs = self._input_size

        # initializes weights and biases
        layer.create_weights(n_inputs, self.generator)
        
        self.layers.append(layer)


    def feed_forward(self, inputs):

        # returns a list of the outputs and activated outputs for each layer

        tmp = inputs
        a_h = [inputs]
        z_h = [inputs]
        for layer in self.layers:

            tmp, z = layer.forward(tmp)
            
            z_h.append(z)
            a_h.append(tmp)
        
        return a_h, z_h


    def feed_forward_out(self, inputs):
        
        # returns the output of the whole network with the current weights

        tmp = inputs
        for layer in self.layers:

            tmp, z = layer.forward(tmp)
        
        return tmp

    def back_prop(self, inputs, targets, learning_rate = 0.1, regularization = 0):

        # performs backpropagation

        for i in range(inputs.shape[0]):
            ins = np.matrix(inputs[i])
            targs = np.matrix(targets[i])

            # get outputs of each layer with feedforward
            a_h, z_h = self.feed_forward(ins)
            
            # perform stochastic gradient descent for each layer going backwards
            prev_layer_err = np.multiply(self.cost_function_grad(targs, a_h[-1]), self.layers[-1]._act_fn_grad(z_h[-1]))

            for j in range(len(self.layers)-1, -1, -1):
                prev_activation_fn = self.layers[j-1 if j > 0 else 0]._act_fn
                prev_layer_err = self.layers[j].backward(a_h[j], z_h[j], prev_layer_err, prev_activation_fn, learning_rate, regularization)
        

    def train(self, inputs, targets, eta= 0.1, epochs = 1000, minibatch_size = 20, regularization= 0):

        # performs training of the model by feedforward and backpropagation iteratively

        minibatch_count = int(inputs.shape[0] / minibatch_size)

        for i in range(1, epochs+1):

            # we permute the data to obtain different minibatches each time
            perm = self.generator.permuted(np.arange(0, inputs.shape[0]))
            inputs = inputs[perm, :]
            targets = targets[perm, :]

            for m in range(minibatch_count):
                idx = minibatch_size * int(self.generator.random() * minibatch_count)
                ins = inputs[idx : idx + minibatch_size]
                targs = targets[idx : idx + minibatch_size]
                self.back_prop(ins, targs, learning_rate=eta, regularization=regularization)