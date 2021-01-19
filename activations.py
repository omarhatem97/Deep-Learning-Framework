import numpy as np
from numpy.core.getlimits import _register_type
from numpy.lib.function_base import kaiser

# tanh activation function 
def tanh(x):
    return np.tanh(x)

# tanh derivative
def tanh_grad(x):
    return 1-np.tanh(x)**2


# ReLU activation function 
def ReLU(x):
    return x * (x > 0)

# ReLU derivative
def ReLU_grad(x):
    return 1 * (x>0)

# sigmoid activation function 
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# sigmoid derivative
def sigmoid_grad(x):
    x = sigmoid(x)
    return x * (1 - x)

# softmax activation function 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x,axis=0, keepdims=True)

def softmax_grad(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


x = np.array([0.00490169, 0.26762315, 0.72747516])




# We don't need soft_max grad, as the error would be calculated before applying the softmax function