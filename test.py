import torch
import layer
import loses
import activations
import numpy as np


x = np.array([[2.25, 0.43, .001], [4.5,3.2,0.121]])
sig  = torch.nn.Softmax(dim=0)
our_sig = activations.softmax(x)
print ( sig(torch.from_numpy(x)))
print ( our_sig)
print ( activations.softmax_grad(our_sig))