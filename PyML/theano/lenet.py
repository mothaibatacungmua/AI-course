import theano
import theano.tensor as T
from net import MLP, train_mlp
import numpy as np
from PyML.libs import softmax_data_wrapper,shared_dat
from layer import LogisticRegression, logistic_error, HiddenLayer, DropoutLayer, ConvPoolLayer

'''
Code based on: http://deeplearning.net/tutorial/
'''

class Lenet(MLP):
    pass

if __name__ == '__main__':
    train, test, validation = softmax_data_wrapper()

    # index to [index]minibatch
    X = T.matrix('X')
    y = T.ivector('y')

    rng = np.random.RandomState(123456)
    batch_size = 500
    classifier = Lenet(X, y, logistic_error, L2_reg=0.0)
    nkernels = [50, 20]
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    classifier.add_layer(ConvPoolLayer(
        filter_shape=(nkernels[0], 1, 5, 5),
        image_shape=(batch_size, 1, 28, 28),
        pool_size=(2,2),
        rng=rng)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    classifier.add_layer(ConvPoolLayer(
        filter_shape=(nkernels[1], nkernels[0], 5, 5),
        image_shape=(batch_size, nkernels[0], 12, 12),
        pool_size=(2, 2),
        rng=rng)
    )

    classifier.add_layer(DropoutLayer(parent=HiddenLayer, p_dropout=0.2, npred=nkernels[1]*4*4, nclass=500, rng=rng))
    classifier.add_layer(DropoutLayer(parent=LogisticRegression, p_dropout=0.3, npred=500, nclass=10))

    train_mlp(shared_dat(train), shared_dat(test), shared_dat(validation), classifier, batch_size=batch_size, learning_rate=0.01)