
## Welcome to Deep-Learning-Framework

This deep learning framework can be used primarily to build deep learning models with ease. it provieds many utilities such as: visualization costs with epochs, Evaluation metrics and more.

### Our project

in this project we implement our Neural Network , a Neural network is simply a function, mapping an input (such as an image) to a prediction. Fundamentally, there are two simple operations you want to do with a function: calculate the output and the derivative, given an input. The first one is needed to obtain predictions, the latter one is to train your network with gradient descent. In neural network terminology, calculating the output is called the forward pass, while the gradient with respect to the input is called a local gradient.

## our modules to build our neural network
- Data Module "loading dataset or split the data"
- core modules
  1- Layers
  2- Losses
  3- Activation functions
- Visualization Module "Visualize the cost function versus iterations/epochs during training process"
- Evaluation Module "Implement accuracy estimation function and implement precision & recall metric for better evaluation"
- Utility Module: "This module is for saving & loading model weights & configurations into a compressed format"

## Dataset :
```markdown
1-Download the dataset of  MNIST as a .csv file from kaggle
2-Create a specific folders for training, validation & testing.
3-Split randomly the training .csv file into training & validation dataset.
```
# First module
## load_data
- input -- file_path, y , indexY
y is the Label
indexY the index of label
return all_pixels, Labels

- input -- image_path 
Take the image and make resize for image then return all pixels in the image

- input -- pixels
Normalize any input by dividing by 255 The max value

- input -- split_dataset(all_pixels, Labels,testSize):
testSize is th size of train 
and Return x_train, y_train, x_tests , y_tests

## build layers
###  Layer
- Initialization for the parameters of the class layer
- input -- the input to the layer for forward propagation.
  return -- computes the output of a layer for a given input
- input -- dY = The gradient of the error with respect to previous layer,
  input -- learning_rate = learning rate to update weights.
  return -- computes the gradient of the error with respect to this layer and update parameters if any.
  
 ###  FC(Layer)

- Initialization for the parameters of the class fully connected layer
        input_size = number of input neurons
        output_size = number of output neurons

- input -- the input to the layer for forward propagation.
        Returns:
        return -- computes the output of a layer for a given input

- input -- dY = The gradient of the error with respect to previous layer,
        input -- learning_rate = learning rate to update weights.
        Returns:
        return -- computes the gradient of the error with respect to this layer and update weights.

### ActivationLayer(Layer)
- input -- activation = pass the name of the activation function,
        input -- learning_rate = pass the name of the activation function gradient.

- input -- the input to the layer for forward propagation.
        Returns:
        return -- computes the output of a layer for a given input

- input -- dY = The gradient of the error with respect to previous layer,
        input -- learning_rate = learning rate to update weights if any.
        Returns:
        return -- computes the gradient of the error with respect to this activation
        
### Flatten(Layer)    
- input -- x = the input from the previous layer,
        Returns:
        return -- changes the shape of the input to flatten the input into one dimension
        extra:
        save -- save the value of the input to use it in reshaping in back propagation.
 - input -- dY = The gradient of the error with respect to previous layer,
        input -- learning_rate = learning rate to update weights if any.
        while we aren't using the input parameters but to follow the notation of backward function in all layers
        Returns:
        return -- changes the shape of the previous layer to be as the input in the forward propagation which stored in input.shape
### Conv_layer(Layer)
- Function to apply one filter to input slice.
        :param input:[numpy array]: slice of input data of shape (f, f, n_C_prev)
        :param W:[numpy array]: One filter of shape (f, f, n_C_prev)
        :param b:[numpy array]: Bias value for the filter. Shape (1, 1, 1)
        :return:
- forward propagation for a 3D convolution layer
        :param X: Input
        :return: Z
-  backward propagation for 3D convlution layer
        :param dY: grad input
        :param learning_rate: the learning rate
        :return: dA
- Apply average pooling.
            Arguments:
            - X: Output of activation function.
            Returns:
            - A_pool: X after average pooling layer
- Distributes error through pooling layer.
            Arguments:
            - dout: Previous layer with the error.
            Returns:
            - dX: Conv layer updated with error.
            
## activation functions
-  tanh activation function
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- activation of x using tanh
- derivative of the tanh function
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- gradient of activation tanh(x) for back propagation
- ReLU activation function
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- activation of x using RelU
- derivative of ReLU activation
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- gradient of activation RelU(x) for back propagation
- sigmoid activation function
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- activation of x using sigmoid
- derivativre of the sigmoid
    x -- Output of the previous layer, of any shape.
    Returns:
    return -- gradient of activation sigmoid(x) for back propagation
- softmax activation function
    x -- Output of the linear layer, of any shape.
    Returns:
    return -- activation of x using softmax
- derivative  of softmax
    x -- Output of the previous layer, of any shape.
    Returns:
    return -- gradient of activation softmax(x) for back propagation
## Losses
- class for losses have two function for every type of loss functions 
    1-Forward function 
    error of X with respect to Y_labels.
    Args:
        X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
        Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
    Returns:
        loss: numpy.mean.float.
    2- Prima function
    differencation of loss function with respect to X at (X, Y).
    Args:
        X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
        Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
    Returns:
        gradX: numpy.ndarray of shape (n_batch, n_dim) which differencation of loss function
- A function to get totel loss by Mean Square Loss (Y - Y_hat)**2
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted  .
            :return : total loss  
- A function to get totel loss by 1/n ( max (0 , - Y * Y_hat)).
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : total loss  
- A function to get totel loss by -log ( | y/2 - 1/2 + WX|).
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : total loss  
- A function to get totel loss by log ( 1+ exp (- Y WX)).
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : total loss  
- A function to get differencation of loss "log ( 1+ exp (- Y WX))".
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : prime of MeanSquareLoss.
- A function to get differencation of max loss as "-Y * Data".
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : prime of MeanSquareLoss.
- A function to get differencation of logIdentity loss as "-Y / 1+e^(-Y Y_hat) * Data".
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : prime of MeanSquareLoss.
- A function to get differencation of logSigmoid loss as "-Y / 1+e^(-Y Y_hat) * Data".
            :param Y_hat: numpy array of Y labeled .
            :param Y    : numpy array of Y predicted .
            :return : prime of MeanSquareLoss.
- A function to get totel loss for softmax layer .
            :param y_hat    : numpy array of Y labeled of data .
            :param y        : numpy array of Y predicted  "2D" for Multiclasses.
            :return : total loss  
- A function to get grad of softmax layer .
            :param  y_hat  : numpy array of Y labeled . 'no_of_sample * no of nodes'
            :param Y_label : numpy array of X is output of final layer. 'no_of_sample * 1'
            :return : numpy array of X is output of grad with dim 'no_of_sample * no of nodes'
## Evaluation metrics          
- calc tn for certain class number
- takes label, predicted_value as vectors
        returns confusion matrix , tp, fp, tn, fn 
- takes label , predicted_value as vectors
        return accuracy
- takes label , predicted_value as vectors
        return precision
- takes label , predicted_value as vectors
        return recall
- takes label , predicted_value as vectors
        return F1_score
## visualizing data
- draw graph between number of epochs on x-axis and losses on y-axis
## save and write model (pickle)
- saves model into a file named 'filename'
- loads a model from a file and returns model
## model
- return losses
- adds layers to the model
        :param layer: a NN layer
- sets the used loss function
        :param loss:
        :param loss_prime:
        :return:
-  predict X for given input
        :param input_data: the input data
        :return:
- train  on sample data
        :param X: data sample
        :param Y: true values
        :param learning_rate: learning rate
- train the model on the dataset
        :param x_train: the training data
        :param y_train: the true values
        :param epochs: number of epochs
        :param learning_rate: the learning rate of the parameters
## utils
- :param shape:
    :return:
- :param shape:
    :param scale:
    :return:
- :param shape:
    :param scale:
    :return:
- A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
- A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
- Function to apply zero padding to the image
        :param X:[numpy array]: Dataset of shape (m, height, width, depth)
        :param pad:[int]: number of columns to pad
        :return:[numpy array]: padded dataset
