
def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: python array (list) containing the dimensions of each layer in our network
    :return:
     parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """


def model_forward(X, paramerters):
    """
     forward propagation
    :param X: data, numpy array of shape (input size, number of examples)
    :param paramerters:output of initialize_parameters
    :return:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """


