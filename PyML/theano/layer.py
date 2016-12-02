import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
from theano.tensor.signal import pool

class Layer(object):
    def __init__(self):
        return NotImplementedError

    def calc_output(self, input):
        return NotImplementedError

    def cost_fn(self, y):
        return NotImplementedError

    def setup(self, input):
        return NotImplementedError

    def feedforward(self, input):
        return NotImplementedError


class LogisticRegression(Layer):
    def __init__(self, npred=1, nclass=1):
        """
        Initialize the parameters of the logistic regression
        :param npred: Number of predictors
        :param nclass: Number of labels need to classify
        """
        self.W = theano.shared(
            value=np.zeros((npred, nclass), dtype=theano.config.floatX),
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros((nclass,), dtype=theano.config.floatX),
            borrow=True

        )

        self.output = None
        self.pred = None
        self.params = [self.W, self.b]
        self.input = None

    def calc_output(self, inp):
        # don't use this line because not robust for infinite
        # https://github.com/Theano/Theano/issues/3162
        # self.softmax_prob = T.nnet.softmax(T.dot(input, self.W) + self.b)
        log_ratio = T.dot(inp, self.W) + self.b
        xdev = log_ratio - log_ratio.max(1, keepdims=True)
        log_softmax = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

        return T.exp(log_softmax)

    def cost_fn(self, X, y):
        return -T.mean(T.log(self.calc_output(X))[T.arange(y.shape[0]), y])
        #return -T.mean(self.log_softmax[T.arange(y.shape[0]),y])

    def predict(self, inp):
        return T.argmax(self.feedforward(inp), axis=1)

    def feedforward(self, inp):
        return self.calc_output(inp)

    def setup(self, inp):
        self.input = inp
        self.output = self.calc_output(inp)

def logistic_error(pred, target):
    return T.mean(T.neq(pred, target))

class HiddenLayer(Layer):
    def __init__(self, npred=1, nclass=1, rng = None, W=None, b=None, activation=T.tanh):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6./ (npred + nclass)),
                    high=np.sqrt(6./ (npred + nclass)),
                    size = (npred, nclass)
                ),
                dtype=theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values,borrow=True)

        if b is None:
            b_values = np.zeros((nclass,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b

        self.activation_fn = activation

        self.output = None
        self.input = None
        self.params = [self.W, self.b]

    def calc_output(self, inp):
        output = T.dot(inp, self.W) + self.b
        output = (output if self.activation_fn is None else self.activation_fn(output))
        return output

    def setup(self, inp):
        self.input = inp
        self.output = self.calc_output(inp)

    def feedforward(self, inp):
        return self.calc_output(inp)

    def predict(self, inp):
        return self.feedforward(inp)


#Creator function
def DropoutLayer(parent=HiddenLayer, p_dropout=0.00, *args, **kwargs):
    class InternalDropoutLayer(parent):
        def __init__(self, p_dropout=0.00, *args, **kwargs):
            parent.__init__(self, *args, **kwargs)
            self.parent = parent
            self.p_dropout = p_dropout

            # Output for testing phase

        def drop(self, X):
            rng = T.shared_randomstreams.RandomStreams(seed=234)
            mask = rng.binomial(n=1, p=1-self.p_dropout, size=X.shape)
            return X * T.cast(mask, theano.config.floatX)

        def calc_output(self, X):
            X_dropout = self.drop(X)
            return self.parent.calc_output(self, X_dropout)

        def feedforward(self, X):
            normal_input = (1 - self.p_dropout)*X
            return self.parent.feedforward(self, normal_input)

    InternalDropoutLayer.__name__ += '_' + parent.__name__
    return InternalDropoutLayer(p_dropout, *args, **kwargs)

class BatchNormalizationLayer(HiddenLayer):
    pass

class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, pool_size=(2,2), rng=None, W=None, b=None):
        '''
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch_size, num_input feature maps, image height, image with)

        :type pool_size: tuple or list of length 2
        :param pool_size: the downsampling (pooling) factor

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        '''

        assert image_shape[1] == filter_shape[1]
        fan_in = np.prod(filter_shape[:1])
        fan_out = np.prod(filter_shape[0] * np.prod(filter_shape[2:]))

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        if not W is None:
            self.W = W
        else:
            self.W = theano.shared(np.asarray(
                rng.uniform(low=-W_bound, high=-W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                borrow=True
            )

        if not b is None:
            self.b = b
        else:
            self.b = theano.shared(
                np.zeros((filter_shape[0],), dtype=theano.config.floatX),
                borrow=True
            )

        self.params = [self.W, self.b]
        self.output = None
        self.input = None
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size

    def calc_output(self, inp):
        conv_out = conv2d(
            input=inp,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.image_shape
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def setup(self, inp):
        self.input = inp
        self.output = self.calc_output(inp)

    def feedforward(self, inp):
        return self.calc_output(inp)