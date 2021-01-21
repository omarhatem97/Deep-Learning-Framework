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


def change_to_vector(y):
    res = []
    for i in range(len(y)):
        curr = y[i]
        res.append(int(np.argmax(curr)))
    return res



if __name__ == '__main__':
    #load train.csv file and save input in x and output in y
    y = 'label'
    indexY = 0
    x, y = load.loadData('DataSet/train.csv', y, indexY)
    x, y, _, _ = load.split_dataset(x, y, 0.01)
    x = load.normalization(x)
    y_matrix = change_to_multi(y)

    #construct layers
    my_model = Model('batch')
    activation_1 = ActivationLayer(ReLU, ReLU_grad)
    activation_2 = ActivationLayer(ReLU, ReLU_grad)

    layer1 = FC(784, 4)
    layer2 = FC(4, 10)
    my_model.add(layer1)
    my_model.add(layer2)
    my_model.add(activation_1)
    my_model.add(activation_2)

    loss = Loss()
    my_model.use(loss= loss.MeanSquareLoss, loss_prime= loss.prime_MeanSquareLoss)

    my_model.fit(x, y_matrix, 100, 0.1)

    #get losses
    losses = my_model.get_losses()

    visualize(losses,100, 0.1)

    y_hat = my_model.predict(x)
    print('yhat :')
    print(y_hat[0].shape)
    print('y :')
    print(y)

    y_hat_vector = change_to_vector(y_hat)
    print('y-hat :')
    print(str(y_hat_vector))





    out, tp, fp, tn, fn = confusion_matrix(y, y_hat_vector, 10)
    print('conf matrix: ')
    print_2dlist(out)

    for i in range(10):
        if(i in y_hat_vector):
            print(str(i) + str(' found!'))
        else:
            print(str(i) + str(' not found!'))




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


