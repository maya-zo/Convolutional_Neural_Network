import numpy as np
from copy import deepcopy
import pickle


'''
functions
'''

def sigmoid(x):  # activation func
    return 1 / (1 + np.exp(-x))
def sigmoid_df(x):  # derivative of activation func
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)
def tanh_df(x):
    return 1-np.square(tanh(x))


def leaky_relu(x):
    return np.maximum(0.01*x, x)
def leaky_relu_df(x):
    r=np.zeros_like(x)
    r[x<=0]=0.01
    r[x>0]=1
    return r


def softmax(x):  # x - datapointsXclasses
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1,keepdims=True)  # softmax, axis 1 because x - datapointsXclasses: out - datapointsXclasses

def softmax_cce(x, y):  # categorical cross entropy loss function on softmax
    x = softmax(x) # x - datapointsXclasses, y - datapointsXclasses
    return -np.sum(y * np.log(x)) / x.shape[0]

def softmax_cce_derivative(x, y):  # categorical cross entropy loss function on softmax derivative
    # x - datapointsXclasses, y - datapointsXclasses
    x = softmax(x)
    d = x - y  # y'i - yi for each i and each datapoint
    return d / x.shape[0]  # axis - datapoints



def softmax_sce(x, y):  # sparse cross entropy loss function on softmax
    # x - datapointsXclasses, y - datapoints
    x = softmax(x)
    return -np.sum(np.log(x[range(x.shape[0]), y])) / x.shape[0]

def softmax_sce_derivative(x, y):  # sparse cross entropy loss function on softmax derivative
    # x - datapointsXclasses, y - datapoints
    x = softmax(x)
    x[range(x.shape[0]), y] -= 1  # y'i - yi for each datapoint, y is index of i: yi =1, otherwise 0 and just y'i
    return x / x.shape[0]  # axis - datapoints



def create_batches(x, y, batch_size,shuff = 0):  # x - num_datasetXpic_size0Xpic_size1Xpic_size2, y - num_datasetX? shuff - want to shuffle or no
    mini_batches = []
    x_size = x.shape
    y_size = y.shape

    x = x.reshape(x_size[0], -1)
    y = y.reshape(y_size[0], -1)
    s = x.shape[1]

    data = np.hstack((x, y))  # merge x and y by axis 1
    if (shuff==1):
        np.random.shuffle(data)  # shuffle by axis 0

    batches_num = data.shape[0] // batch_size + (1 if data.shape[0] % batch_size > 0 else 0)  # num of batches

    for i in range(batches_num):
        size = min((i + 1) * batch_size, data.shape[0]) - (i * batch_size)
        mini_batch = data[(i * batch_size):((i * batch_size) + size)]  # making batch

        X_mini = mini_batch[:, :s]  # dividing to x and y
        X_mini = X_mini.reshape(size,*x_size[1:])

        Y_mini = mini_batch[:, s:]
        Y_mini = Y_mini.reshape(size,*y_size[1:])

        Y_mini = Y_mini.astype(int)
        mini_batches.append((X_mini, Y_mini))  # appending batch to mini_batches

    return mini_batches  # list of tuples length 2 of numpy arrays, axis 0 - num_dataset.


'''
classes
'''


class Layer:
    def __init__(self):
        self.input = None
        self.out = None

    def forward(self, inputs):  # computes the output Y of a layer for a given input X
        pass

    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):  # computes dE/dX for a given dE/dY (and update parameters if any)
        pass


class fcLayer(Layer):
    def __init__(self, inSize, outSize, w_init = None): #w_inits = option for self initialization of weights and bias. mainly for testing, w_init = (weights,bias)
        super().__init__()
        print("fc: ",inSize,"->",outSize)
        if w_init ==None:
            self.weights = 2*(np.random.rand(inSize, outSize)-0.5)*np.sqrt(6/(inSize+outSize))
            self.bias = np.ones([1, outSize])* 0.01
        else:
            self.weights = w_init[0]
            self.bias = w_init[1]

        #adam additions
        self.mWeights = np.zeros([*self.weights.shape]) #adam: first momentum - aggravated mean of gradients
        self.vWeights = np.zeros([*self.weights.shape]) #adam: second momentum - aggravated variance of gradients

        self.mBias = np.zeros([*self.bias.shape]) #adam: first momentum - aggravated mean of gradients
        self.vBias = np.zeros([*self.bias.shape]) #adam: second momentum - aggravated variance of gradients


    def forward(self, inputs):
        print("fc")
        self.input = inputs  # datapointsXinSize
        self.out = np.dot(self.input, self.weights) + self.bias
        return self.out  # datapointsXoutsize

    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):  # calc gradients
        print("fc_back")
        Dweights = None
        if (self.input.shape[0] == 1): #datapoints = 1
            Dweights = np.dot(self.input.reshape((self.input.T.shape[0],-1)), Dout)
        else:
            Dweights = np.dot(self.input.T,Dout)  # Dout = datapointsXoutSize, input.T = inSizeXdatapoints, weight =  inSizeXoutSize
        Dbias = np.sum(Dout, axis=0)  # axis 0 - datapoints
        Din = np.dot(Dout,self.weights.T)  # for prev level, Dout = datapointsXoutSize, weights.T = outSizeXinSize: Din - datapointsXinSize

        if not adam: #regular sgd
            self.weights -= learnRate * Dweights  # update weights - for grad descent
            self.bias -= learnRate * Dbias
        else: #adam
            curr_beta = (beta[0]**(epochNum+1), beta[1]**(epochNum+1))  # beta(t) = beta^t
            #update momentums
            self.mWeights= beta[0]*self.mWeights + (1-beta[0])*Dweights
            self.vWeights = beta[1] * self.vWeights + (1 - beta[1])*(Dweights**2)
            self.mBias = beta[0] * self.mBias + (1 - beta[0])*Dbias
            self.vBias = beta[1] * self.vBias + (1 - beta[1])*(Dbias**2)

            # correcting bias - so wont be to close to zero
            mW_hat = self.mWeights/(1-curr_beta[0])
            vW_hat = self.vWeights / (1 - curr_beta[1])
            mB_hat = self.mBias/(1-curr_beta[0])
            vB_hat = self.vBias / (1 - curr_beta[1])

            # update weights
            self.weights -= learnRate * mW_hat/np.sqrt(vW_hat+eps)
            self.bias -= learnRate * mB_hat / np.sqrt(vB_hat + eps)

        return Din


class ActivationLayer(Layer):
    def __init__(self, activationFunc, activationDerivative):
        super().__init__()
        print("Activation:",activationFunc.__name__ )
        self.activation = activationFunc
        self.derivative = activationDerivative

    def forward(self, inputs):
        print("activation")
        self.input = inputs  # datapointsXSize
        self.out = self.activation(self.input)
        return self.out

    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):
        print("activation_back")
        return self.derivative(self.input) * Dout  # for prev level , Dout = datapointsXSize


class ConvLayer(Layer):
    def __init__(self, filters_num, size, w_init=None): #w_init - same as fc layer
        super().__init__()
        print("conv: filters- ", filters_num, " size- ", size)
        self.filters = filters_num  # f
        self.kernel_size = size  # h X w X c

        if w_init ==None:
            self.bias = np.ones(self.filters) * 0.01
            self.weights = 2*(np.random.rand(*size, self.filters)-0.5)*np.sqrt(6/(np.prod(self.kernel_size)+self.filters))  # h X w X c X f
        else:
            self.weights = w_init[0]
            self.bias = w_init[1]


        # adam additions
        self.mWeights = np.zeros([*self.weights.shape])  # adam: first momentum - aggravated mean of gradients
        self.vWeights = np.zeros([*self.weights.shape])  # adam: second momentum - aggravated variance of gradients

        self.mBias = np.zeros([*self.bias.shape])  # adam: first momentum - aggravated mean of gradients
        self.vBias = np.zeros([*self.bias.shape])  # adam: second momentum - aggravated variance of gradients



    def forward(self, inputs):
        print("conv")
        self.input = inputs  # datapoints X h X w X c
        ax0 = self.input.shape[0]  # datapoints
        ax1 = self.input.shape[1] - self.kernel_size[0] + 1  # N-F/S +1 , S =1
        ax2 = self.input.shape[2] - self.kernel_size[1] + 1
        ax3 = self.filters  # new num of channels = prev num of filters
        self.out = np.zeros([ax0, ax1, ax2, ax3])

        for x in range(ax0):
            for f in range(self.filters):
                for i in range(ax1):
                    for j in range(ax2):
                        matrix = self.input[x, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :]
                        self.out[x, i, j, f] = np.sum(matrix * self.weights[:, :, :, f]) + self.bias[f]
        return self.out



    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):
        print("conv_back")
        # Dout sizes:
        ax0 = self.input.shape[0]  # datapoints
        ax1 = self.input.shape[1] - self.kernel_size[0] + 1  # N-F/S +1 , S =1
        ax2 = self.input.shape[2] - self.kernel_size[1] + 1
        ax3 = self.filters  # new num of channels = prev num of filters

        Din = np.zeros([*self.input.shape])
        Dweights = np.zeros([*self.weights.shape])
        Dbias = np.zeros([self.filters])

        for x in range(ax0):
            for f in range(ax3):
                for i in range(ax1):
                    for j in range(ax2):
                        Dbias[f] += Dout[x, i, j, f]
                        Din[x, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :] += Dout[x, i, j, f] * self.weights[:, :, :, f]
                        Dweights[:, :, :, f] += self.input[x, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]),:] * Dout[x, i, j, f]

        if not adam: #regular sgd
            self.weights -= learnRate * Dweights  # update weights - for grad descent
            self.bias -= learnRate * Dbias
        else: #adam
            curr_beta = (beta[0]**(epochNum+1), beta[1]**(epochNum+1))  # beta(t) = beta^t
            #update momentums
            self.mWeights= beta[0]*self.mWeights + (1-beta[0])*Dweights
            self.vWeights = beta[1] * self.vWeights + (1 - beta[1])*(Dweights**2)
            self.mBias = beta[0] * self.mBias + (1 - beta[0])*Dbias
            self.vBias = beta[1] * self.vBias + (1 - beta[1])*(Dbias**2)
            # correcting bias - so wont be to close to zero
            mW_hat = self.mWeights/(1-curr_beta[0])
            vW_hat = self.vWeights / (1 - curr_beta[1])
            mB_hat = self.mBias/(1-curr_beta[0])
            vB_hat = self.vBias / (1 - curr_beta[1])
            # update weights
            self.weights -= learnRate * mW_hat/np.sqrt(vW_hat+eps)
            self.bias -= learnRate * mB_hat / np.sqrt(vB_hat + eps)

        return Din


class MaxPoolLayer(Layer):
    def __init__(self, size=(2, 2)):
        super().__init__()
        print("maxpool: ", "size- ", size)
        self.size = size  # size is (h,w)

    def forward(self, inputs):
        print("maxpool")
        self.input = inputs  # datapoints X h X w X c
        ax0 = self.input.shape[0]
        ax1 = self.input.shape[1] // self.size[0]
        ax2 = self.input.shape[2] // self.size[1]
        ax3 = self.input.shape[3]
        self.out = np.zeros([ax0, ax1, ax2, ax3])

        self.index0 = np.zeros([ax0, ax1, ax2, ax3],dtype = int)
        self.index1 = np.zeros([ax0, ax1, ax2, ax3],dtype = int)

        for x in range(ax0):
            for f in range(ax3):
                for i in range(ax1):
                    for j in range(ax2):
                        curr = self.input[x, i * self.size[0]:(i + 1) * self.size[0], j * self.size[1]:(j + 1) * self.size[1], f]
                        max = np.unravel_index(curr.argmax(), curr.shape)
                        self.index0[x, i, j, f] = i * self.size[0] + max[0]
                        self.index1[x, i, j, f] = j * self.size[1] + max[1]
                        self.out[x, i, j, f] = self.input[x, self.index0[x, i, j, f], self.index1[x, i, j, f], f]

        return self.out

    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):
        print("maxpool_back")
        ax0 = self.input.shape[0]
        ax1 = self.input.shape[1] // self.size[0]
        ax2 = self.input.shape[2] // self.size[1]
        ax3 = self.input.shape[3]

        Din = np.zeros([*self.input.shape])
        for x in range(ax0):
            for f in range(ax3):
                for i in range(ax1):
                    for j in range(ax2):
                        Din[x, self.index0[x, i, j, f], self.index1[x, i, j, f], f] = Dout[x, i, j, f]
        return Din


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        print("flatten")

    def forward(self, inputs):
        print("flatten")
        self.input = inputs  # datapointsX(...) - nd
        self.out = self.input.reshape(self.input.shape[0], -1)
        return self.out  # datapointsXsize - 2d

    def backward(self, Dout, learnRate=0.001,adam = False,beta=(0.9,0.999),eps = 0.00000001,epochNum=0):
        print("flatten_back")
        Din = Dout.reshape(*self.input.shape)
        return Din  # datapointsX(...) - input size



class Network:
    def __init__(self, lossFunc, DlossFunc):
        self.loss = lossFunc
        self.lossDerivative = DlossFunc
        self.layers = []
        self.lossList = []
        self.totalLossList = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, epochs_num = 20, batch_size = 128,learnRate = 0.001,path = None,adam = False,val = None,beta =(0.9,0.999),eps=0.00000001):  #x - num_datasetX pic_size0 X pic_size1 X pic_size2 , y - num_dataset.
        for i in range(epochs_num):
            totalLoss = 0
            count = 0
            batches = create_batches(x, y, batch_size,1)
            totalAcc = 0
            for batch in batches:
                count += 1
                output = batch[0]  # batch[0] = x
                # forward propagation
                for layer in self.layers:
                    output = layer.forward(output)
                #calc accuracy
                acc = np.mean(softmax(output).argmax(axis=1) == batch[1])
                totalAcc+=acc
                #calc loss
                loss = self.loss(output, batch[1])  # batch[1] = x
                self.lossList.append(loss)
                totalLoss += loss
                print("forward: batch: ", count, "   loss: ", loss, "  accuracy: ",acc)

                # back propagation
                derivative = self.lossDerivative(output, batch[1])
                for layer in reversed(self.layers):
                    derivative = layer.backward(derivative,learnRate,adam,beta,eps,i)

                print("backward: batch: ", count)

            self.totalLossList.append(totalLoss / count)
            if path!=None:
                f = open(path + str(i), "wb")
                pickle.dump(deepcopy(self), f)
                f2 = open(path + str(i)+"loss", "wb")
                pickle.dump((deepcopy(self.lossList),deepcopy(self.totalLossList)), f2)

            print("epoch: ", i, " average loss: ", totalLoss/count,  " average acc: ", totalAcc/count)
            if val!=None:
                prediction = self.predict(val[0])[0]
                val_accuracy = np.mean(prediction == val[1])
                print("val accuracy: ",val_accuracy)

        return self



    def predict(self, inputs, batchSize =1000): #returns (highest val index (prediction index),output arr(percentages))
        maxIndex = np.zeros(1)
        y = np.zeros(inputs.shape)
        batches = create_batches(inputs, y, batchSize)
        for batch in batches:
            output = batch[0]
            for layer in self.layers:
                output = layer.forward(output)

            output = softmax(output)
            #print(output)
            #print("argmax ",output.argmax(axis=1))
            maxIndex = np.concatenate((maxIndex,output.argmax(axis=1)))
        maxIndex = np.delete(maxIndex, 0, axis=0)
        return (maxIndex, output)





