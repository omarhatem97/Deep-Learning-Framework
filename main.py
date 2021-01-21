
from numpy.core.defchararray import mod


from activations import *
from Evaluation import *
from layer import *
from Load import *
from loss import *
from model import *
from save_and_load import *
from utils import *
from visualization import *


def change_to_multi (y):
    """takes y as vector , returns result as matrix"""
    res = np.zeros((len(y), np.max(y) + 1))
    res[np.arange(len(y)), y] = 1
    return res





#load train.csv file and save input in x and output in y
#y = 'label'
indexY = 0
#x, y = load.loadData('DataSet/train.csv', y, indexY)
#y = change_to_multi(y)


x = np.random.rand(11,13,13,256) / 100.0
y = np.random.rand(11,5) / 100.0

#construct layers
my_model = Model('batch')

l = Loss()

layer2 = Conv_layer(filters=10,kernel_shape=(3,3),padding='same',stride=1)
my_model.add(layer2)

layer4= Pool(stride=2,filter_size=6, mode='max')
my_model.add(layer4)
layer3 = Flatten()

my_model.add(layer3)

layer1 = FC(5)

activation1 = ActivationLayer(sigmoid, sigmoid_grad)
my_model.add(layer1)
my_model.add(activation1)
my_model.use(loss= l.MeanSquareLoss, loss_prime= l.prime_MeanSquareLoss)

my_model.fit(x, y, 10, 0.1)
# arr = [1,5,3]
# print(str(change_to_multi(arr)))

def change_to_vector(y):
    res = []
    for i in range(len(y)):
        curr = y[i]
        res.append(int(np.argmax(curr)))
    return res



out, tp, fp, tn, fn = confusion_matrix(y, y_hat_vector, 10)
print('conf matrix: ')
print_2dlist(out)

print('tp: ' + str(tp))
print('tn: ' + str(tn))
print('fp: ' + str(fp))
print('fn: ' + str(fn))
print('accuracy: ' + str(accuracy(y,y_hat_vector,10)))
print('precision: ' + str(precision(y, y_hat_vector,10)))
print('recall: ' + str(recall(y, y_hat_vector, 10)))
print('F1 score: ' + str(F1_score(y, y_hat_vector,10)))







# arr = [1,5,3]
# print(str(change_to_multi(arr)))


