import theano.tensor as T
import numpy as np
from net import MLP, train_mlp
from layer import LogisticRegression, logistic_error,DropoutLayer,HiddenLayer
from PyML.libs import softmax_data_wrapper,shared_dat


'''
Code based on: http://deeplearning.net/tutorial/
'''


if __name__ == '__main__':
    train, test, validation = softmax_data_wrapper()

    # index to [index]minibatch
    X = T.matrix('X')
    y = T.ivector('y')

    rng = np.random.RandomState(123456)
    classifier = MLP(X, y, logistic_error, L2_reg=0.0)
    #classifier.add_layer(HiddenLayer(28*28, 100, rng=rng))
    #classifier.add_layer(HiddenLayer(100, 60, rng=rng))
    #classifier.add_layer(LogisticRegression(60, 10))

    classifier.add_layer(DropoutLayer(parent=HiddenLayer, p_dropout=0.1, npred=28 * 28, nclass=100, rng=rng))
    classifier.add_layer(DropoutLayer(parent=HiddenLayer, p_dropout=0.2, npred=100, nclass=60, rng=rng))
    classifier.add_layer(DropoutLayer(parent=LogisticRegression, p_dropout=0.3, npred=60, nclass=10))

    train_mlp(shared_dat(train), shared_dat(test), shared_dat(validation), classifier)
