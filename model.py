import numpy as np 
class Model:
    def __init__(self, training_method):
        self.training_method = training_method
        self.layers = []
        self.err = 0
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict X for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            X = input_data[i]
            for layer in self.layers:
                X = layer.forward(X)
            result.append(X)

        return result

    def train(self,X,Y, learning_rate):
        for layer in self.layers:
            X = layer.forward(X)

        # compute loss (for display purpose only)
        self.err += self.loss(Y, X)

        # backward propagation
        error = self.loss_prime(Y, X)
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)




    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            self.err = 0
            if(self.training_method =='online'):
                for j in range(samples):
                    # forward propagation
                    X = x_train[j,:].reshape(1,x_train.shape[1])
                    Y = y_train[j,:].reshape(1,y_train.shape[1])
                    self.train(X,Y,learning_rate)
                    
            elif (self.training_method =='batch'):
                   batch_size = 100
                   num_of_batches = max(1, samples/batch_size)
                   j = 0
                   for j in range(num_of_batches):
                       begin_index = j*batch_size
                       end_index = min (samples, begin_index+batch_size)
                       current_batch_size = end_index-begin_index
                       X = x_train[begin_index:end_index , :].reshape(current_batch_size,x_train.shape[1])
                       Y = y_train[begin_index:end_index , : ].reshape(current_batch_size,y_train.shape[1])
                       self.train(X,Y,learning_rate)



            # calculate average error on all samples
            self.err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, self.err))