import numpy as np
import math
'''
1-Forward function 
error of X with respect to Y_labels.
Args:
    X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
    Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
Returns:
    loss: numpy.float. Mean square error of x.
2- Local gradient function
Local gradient with respect to X at (X, Y).
Args:
    X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
    Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
Returns:
    gradX: numpy.ndarray of shape (n_batch, n_dim) which delta
'''

'''
the types of loss function : 
1/n  (( WX - Y )**2)
1/n ( max (0 , - Y WX))
1/n ( max (0 , 1 - Y WX))
log ( 1+ exp (- Y WX))
- log ( | y/2 - 1/2 + WX)
'''
 
class Loss:
    def forward_MeanSquareLoss(self, X, Y):
        # calculating loss of (y_hat - y_labeled)**2

        differences = X - Y
        differences_squared = differences ** 2
        Msq_loss = np.mean(differences_squared)
        # Msq_loss = np.sqrt(Msq_loss)
        return Msq_loss

    def local_grad_MeanSquareLoss(self, X, Y):
        rolls = np.zeros(shape= ())
        # grads = {'X': 2 * ((np.subtract(X,Y)) / X.shape[0]}
        grads =  2 * ((np.subtract(X,Y)) / len(X))
        rolls = grads
        return rolls

    def forward_CrossEntropy(self, X, y):
        exp_x = np.exp(X)
        probs = exp_x/np.sum(exp_x, axis=1, keepdims=True)
        log_probs = -1 * np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(log_probs)

        # # caching for backprop
        # self.cache['probs'] = probs
        # self.cache['y'] = y

        return crossentropy_loss

    def local_grad_CrossEntropy(self, X, Y , probs):
        # probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {'X': (probs - ones)/float(len(X))}
        return grads

    def forward_log_loss(self , X , Y):
        #handle -ve with log 
        differences = ( Y / 2) + 0.5 + X
        loss_li = -1 * np.log(differences) 
        print ( loss_li)
        log_loss = np.mean(loss_li)
        return log_loss
    

    def forward_max(self , X , Y):
        cal_arr = np.array(-1 * X * Y)
        for i in cal_arr:
            if i < 0 :
                i = 0
        max_loss = np.mean(cal_arr)
        return max_loss



# y_hat = np.array([0.000, 0.166, 0.333])
# y_true = np.array([0.000, 0.254, 0.998])

# a = Loss()
# print ( a.forward_max(y_hat , y_true))