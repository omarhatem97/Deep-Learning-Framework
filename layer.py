import numpy as np


# Base class for Layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, dY, learning_rate):
        raise NotImplementedError


class FC(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = 0.1 * np.random.rand(input_size, output_size) 
        self.bias = 0.1 * np.random.rand(1, output_size) 

    # returns output for a given input
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given dY=dE/dY. Returns grad=dE/dX.
    def backward(self, dY, learning_rate):
        m = len(dY)
        grad = np.dot(dY, self.weights.T)
        dW = np.dot(self.input.T, dY)/m
        dB = np.squeeze(np.sum(dY, axis=0, keepdims=True)).reshape(1,dY.shape[1])/m

        # update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB
        return grad

class ActivationLayer(Layer):
    def __init__(self, activation, activation_grad):
        self.activation = activation
        self.activation_grad = activation_grad

    # returns the activated input
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns grad=dE/dX for a given dY=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, dY, learning_rate):    
        return self.activation_grad(self.input) * dY

class Flatten(Layer):
   
    def forward(self, X):
        self.input = X
        samples = X.shape[0]
        return X.reshape(samples, -1)
        
    def backward(self, dY, learning_rate):
        return dY.reshape(self.input.shape)

