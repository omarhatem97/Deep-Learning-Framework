import layer
from activations import *
import model
import Load
from loss import *
train_data = Load.load.loadData('DataSet/train.csv','label', 0)

print(train_data[0].shape)

X = train_data[0]
Y = train_data[1]

X = X.reshape(42000,28,28,1)
Y = Y.reshape(42000,1)

print(X.shape, Y.shape)

our_model = model.Model('batch')

# Alex-net Maybee??
# layer1 = layer.Conv_layer(6, kernel_shape=(5,5),padding='same',stride=1)
# layer2 = layer.Pool(filter_size=3,stride=2,mode='max')
# layer3 = layer.Conv_layer(256, kernel_shape=(5,5),padding='same',stride=4)
# layer4 = layer.Pool(filter_size=3,stride=2,mode='max')
# layer5 = layer.Conv_layer(384, kernel_shape=(3,3),padding='same',stride=4)
# layer6 = layer.Conv_layer(384, kernel_shape=(3,3),padding='same',stride=4)
# layer7 = layer.Conv_layer(256, kernel_shape=(3,3),padding='same',stride=4)
# layer8 = layer.Pool(filter_size=3,stride=2,mode='max')
# layer9 = layer.Flatten()
# layer10 = layer.FC(4096)
# layer11 = layer.FC(4096)

layer1 = layer.Conv_layer(6, kernel_shape=(5,5),padding='same',stride=1)
activation1= layer.ActivationLayer(ReLU, ReLU_grad)
layer2 = layer.Pool(filter_size=2,stride=2,mode='average')
layer3 = layer.Conv_layer(16, kernel_shape=(5,5),padding='same',stride=1)
activation2= layer.ActivationLayer(ReLU, ReLU_grad)
layer4 = layer.Pool(filter_size=2,stride=2,mode='average')
layer5 = layer.Flatten()
layer6 = layer.FC(120)
layer7 = layer.FC(84)
layer8 = layer.FC(10)
activation3= layer.ActivationLayer(softmax, softmax_grad)

our_model.add(layer1)
our_model.add(activation1)
our_model.add(layer2)
our_model.add(layer3)
our_model.add(activation2)
our_model.add(layer4)
our_model.add(layer5)
our_model.add(layer6)
our_model.add(layer7)
our_model.add(layer8)
our_model.add(activation3)

our_loss = Loss()
our_model.use(our_loss.forward_CrossEntropy, our_loss.prime_CrossEntropy )



our_model.fit(X,Y,10,0.1)

print(X.shape)

our_model.predict(X)

