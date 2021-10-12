import numpy as np
import NNetwork as n
import glob
from PIL import Image
import pickle
import matplotlib.pyplot as plt

#constants:
datasetPath = "C:\\Users\\user\\Documents\\SCHOOL\\DS\\project\\dataset"
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise" ]
imageSize = (48,48,1)
learnRate = 0.001
batchSize = 128
epoch_num = 20
modelPath = "C:\\Users\\user\\Documents\\SCHOOL\\DS\\project\\saved_modules2\\adam7"  #model path that we want to open for more train/ for testing
newmodelPath = "C:\\Users\\user\\Documents\\SCHOOL\\DS\\project\\saved_modules3\\adam" #path to save model after train



def getData(path,classList,imgSize): #opens dataset and organizes data as Datapoints X img_h X img_w X 1
    test=np.zeros([1, *imgSize])
    test_y=np.zeros(1)
    train=np.zeros([1, *imgSize])
    train_y=np.zeros(1)

    for c in range(len(classList)):
        print("*")
        curr_path_train = path + "\\train\\" + classList[c]
        count=0
        data = np.zeros([1, 48, 48, 1])
        for i in glob.glob(curr_path_train +'**/*.jpg'):
            count +=1
            image = Image.open(i)
            x = np.asarray(image).reshape(1,*imgSize)  # 1X48X48X1 numpy array
            data=np.concatenate((data,x)) #adding img

        #count = np.min([count,4000]) #makes sure that no class has more than 4000 datapoints
        data = np.delete(data, 0, axis=0) #deleting zero row
        print("c: ", classes[c], " x:", data[:count].shape, " y: ", count)
        train = np.concatenate((train,data[:count])) #adding class to train
        train_y = np.concatenate( ( train_y,(np.ones(count)*c) ) ) #adding c label

        #now for test
        curr_path_test = path + "\\test\\" + classList[c]
        count = 0
        data = np.zeros([1, 48, 48, 1])
        for i in glob.glob(curr_path_test +'**/*.jpg'):
            count +=1
            image = Image.open(i)
            x = np.asarray(image).reshape(1,*imgSize)  # 1X48X48X1 numpy array
            data=np.concatenate((data,x)) #adding img
        data = np.delete(data, 0, axis=0) #deleting zero row
        test = np.concatenate((test,data)) #adding class to train
        test_y = np.concatenate( ( test_y,(np.ones(count)*c) ) ) #adding c label

    test = np.delete(test, 0, axis=0)
    test_y = np.delete(test_y, 0, axis=0)
    train = np.delete(train, 0, axis=0)
    train_y = np.delete(train_y, 0, axis=0)

    test = test/255 #normalizing
    train = train/255 #normalizing
    print("train: ", train.shape, train_y.shape)
    print("test: ", test.shape, test_y.shape)

    return((train,train_y),(test,test_y))


def getTest(path, classList, imgSize): #like get data but only test
    test = np.zeros([1, *imgSize])
    test_y = np.zeros(1)

    for c in range(len(classList)):
        print("*")
        # now for test
        curr_path_test = path + "\\test\\" + classList[c]
        count = 0
        data = np.zeros([1, 48, 48, 1])
        for i in glob.glob(curr_path_test + '**/*.jpg'):
            count += 1
            image = Image.open(i)
            x = np.asarray(image).reshape(1, *imgSize)  # 1X48X48X1 numpy array
            data = np.concatenate((data, x))  # adding img
        data = np.delete(data, 0, axis=0)  # deleting zero row
        test = np.concatenate((test, data))  # adding class to train
        test_y = np.concatenate((test_y, (np.ones(count) * c)))  # adding c label

    test = np.delete(test, 0, axis=0)
    test_y = np.delete(test_y, 0, axis=0)

    test = test / 255 #normalizing
    print("test: ", test.shape, test_y.shape)

    return (test, test_y)


def openPickle(path):
    f = open(path,"rb")
    return pickle.load(f)


def build():
    net = n.Network(n.softmax_sce, n.softmax_sce_derivative)  # network with cross entropy loss and softmax
    net.add(n.ConvLayer(16, (5, 5, 1)))  # 44x44x32
    net.add(n.ActivationLayer(n.leaky_relu, n.leaky_relu_df))
    net.add(n.MaxPoolLayer((2, 2)))  # 22x22x64
    net.add(n.ConvLayer(32, (3, 3, 16)))  # 20x20x32
    net.add(n.ActivationLayer(n.leaky_relu, n.leaky_relu_df))
    net.add(n.ConvLayer(16, (5, 5, 32)))  # 16x16x32
    net.add(n.ActivationLayer(n.leaky_relu, n.leaky_relu_df))
    net.add(n.MaxPoolLayer((2, 2)))  # 8x8x16
    net.add(n.Flatten())
    net.add(n.fcLayer(1024, 256))
    net.add(n.ActivationLayer(n.tanh, n.tanh_df))
    net.add(n.fcLayer(256, 64))
    net.add(n.ActivationLayer(n.tanh, n.tanh_df))
    net.add(n.fcLayer(64, 7))

    return net


def trainModel(net,train,test,path):
    print("----starting training-----")
    net.train(train[0], train[1], epoch_num, batchSize, learnRate ,path, False, test)
    return net


def loss_graph(path):
    loss,total= openPickle(path+"loss")
    print(total)
    plt.plot(total)
    plt.show()

def testing(net, test):
    prediction = net.predict(test[0])

    # accuracy
    acc = np.mean(prediction[0] == test[1])
    print("accuracy: ", acc)
    #precision - recall
    precision = np.zeros(7)
    recall = np.zeros(7)
    for c in range(len(classes)):
        x = np.count_nonzero(prediction[0] == c)
        if (x == 0):
            precision[c] = -1
        else:
            precision[c] = np.count_nonzero((prediction[0] == c) & (test[1] == c)) / x
        recall[c] = np.count_nonzero((prediction[0] == c) & (test[1] == c)) / np.count_nonzero(test[1] == c)
    print("precision: ", precision)
    print("recall: ", recall)
    return precision,recall



if __name__ == "__main__":

    '''
    If I wanted to train:
    first get data:
        train, test = getData(datasetPath, classes, imageSize)
    
    to build a model:
        net = build()
    or to open one:
        net = openPickle(modelPath)
        
    and to train it:
        trainModel(net,train,test,newmodelPath)
    '''

    
    #testing
    test = getTest(datasetPath, classes, imageSize)
    net = openPickle(pt)
    testing(net, test)
    loss_graph(pt)








