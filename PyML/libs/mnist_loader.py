import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

mnist_train = None
mnist_validation = None
mnist_test = None


def load_data():
    global mnist_train, mnist_validation, mnist_test

    if mnist_train is None:
        f = gzip.open('../data/mnist.pkl.gz')
        mnist_train, mnist_validation, mnist_test = cPickle.load(f)
        f.close()

    return (mnist_train, mnist_validation, mnist_test)


def display_image(index):
    train, validation, test = load_data()

    pixels_arr = train[0][index]

    mat = pixels_arr.reshape(28, 28)
    plt.figure(figsize=(3,3))
    #plt.imshow(mat, cmap='gray')
    plt.imshow(mat, cmap=matplotlib.cm.binary)
    plt.show()


def neunet_data_wrapper():
    train, validation, test = load_data()

    train_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
    train_outputs = [vectorized_result(y) for y in train[1]]
    train_data = zip(train_inputs, train_outputs)

    validation_inputs = [np.reshape(x, (784,1)) for x in validation[0]]
    validation_outputs = [vectorized_result(y) for y in validation[1]]
    validation_data = zip(validation_inputs, validation_outputs)

    test_inputs = [np.reshape(x, (784,1)) for x in test[0]]
    test_outputs = [vectorized_result(y) for y in test[1]]
    test_data = zip(test_inputs, test_outputs)

    return (train_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def softmax_data_wrapper():
    pass

def svm_data_wrapper():
    pass

'''
if __name__ == '__main__':
    display_image(10)
    pass
'''