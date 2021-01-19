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
2- Prima function
differencation of loss functionwith respect to X at (X, Y).
Args:
    X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
    Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
Returns:
    gradX: numpy.ndarray of shape (n_batch, n_dim) which differencation of loss function
'''

'''
the types of loss function : 
MeanSquareLoss = 1/n  (( WX - Y )**2)
1/n ( max (0 , - Y WX))
log ( 1+ exp (- Y WX))
- log ( | y/2 - 1/2 + WX|)
1/n ( max (0 , 1 - Y WX))
−(ylog(X)+(1−y)log(1−X))
'''
 
class Loss:
    def MeanSquareLoss(self, Y_hat , Y):
        # (Y - Y_hat)**2
        Msq_loss = np.mean((Y_hat - Y)**2)
        return Msq_loss

    def max_Loss(self , Y_hat , Y):
        #1/n ( max (0 , - Y * Y_hat))
        cal_arr = np.array(-1 * Y_hat * Y)
        for i in range(len(cal_arr)):
            if cal_arr[i] < 0 :
                cal_arr[i] = 0
        max_loss = (-1 * np.mean(cal_arr))
        return max_loss

    def forward_log_sigmoid_loss(self , X , Y):
        #- log ( | y/2 - 1/2 + WX|)
        differences = ( Y / 2) - 0.5 + X
        for i in range(len(differences)):
            if differences [ i ] < 0:
                differences [ i ] *= -1
        loss_li = -1 * np.log(differences) 
        log_loss = np.mean(loss_li)
        return log_loss

    def forward_log_identity_loss(self , X , Y):
        #log ( 1+ exp (- Y WX))
        differences = 1 + np.exp(-1*Y*X)
        for i in range(len(differences)):
            if differences [ i ] < 0:
                differences [ i ] *= -1
        loss_li =  np.log(differences) 
        log_loss = np.mean(loss_li)
        return log_loss

    # def hinge_loss_single( self , Y_hat , Y ):
    #     #1/n ( max (0 , - Y * Y_hat))
    #     cal_arr = np.array(1 - (1 * Y_hat * Y))
    #     for i in range(len(cal_arr)):
    #         if cal_arr[i] < 0 :
    #             cal_arr[i] = 0
    #     print(cal_arr)
    #     max_loss = ( np.mean(cal_arr))
    #     return max_loss
    
    def prime_MeanSquareLoss(self , Y_hat , Y):
        # (2 * (Y_hat - Y )) * 
        grads = (2 * (Y_hat - Y ))/ len(Y)
        return grads
    
    def local_grad_maxLoss(self , Y ):
        # -Y * 
        grads = (-1 * Y ) / len(Y)
        return grads

    def local_grad_logIdentityLoss(self , Y , Y_hat ):
        # -Y / 1+e^(-Y Y_hat) * 
        grads = (-1 * Y * np.exp(-1*Y*Y_hat) ) / ((1 + np.exp(-1*Y*Y_hat))*len(Y))
        return grads
    
    def local_grad_logSigmoidLoss(self , Y , Y_hat ):
        # -Y / 1+e^(-Y Y_hat) * 
        grads = (-1 * Y ) / (1 + np.exp(-1*Y*Y_hat))
        return grads

    def forward_CrossEntropy(self, X, y):
        exp_x = np.exp(X)
        probs = exp_x/np.sum(exp_x , axis=1 , keepdims=True)
        log_probs = -1 * np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(log_probs)
        return crossentropy_loss 

    def local_grad_CrossEntropy(self, X, Y):
        exp_x = np.exp(X)
        probs = exp_x/np.sum(exp_x, axis=1, keepdims=True)
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads =  (probs - ones)/float(len(X))
        return grads

# predictions = np.array([-1,0.25,0.25,-0.25])
# targets = np.array([-1,0,0,1])

# a = Loss()
# print ( a.hinge_loss_single(predictions , targets))
