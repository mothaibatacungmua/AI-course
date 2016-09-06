import theano
import sys
import numpy as np
from theano import tensor as T


class QValueNet(object):
    def __init__(self, game_size):
        self.train = None
        self.game_size = game_size

        X = T.vector()
        y = T.vector()

        self.w1 = self.init_weights((self.game_size ** 2, 10), "w1")
        self.w2 = self.init_weights((10, 20), "w2")
        self.wo = self.init_weights((20, 2), "wo")

        t = self.feedforward(X, self.w1, self.w2, self.wo)
        cost = T.mean(T.sqr(t - y))
        params = [self.w1, self.w2, self.wo]
        updates = self.RMSprop(cost, params)

        self.train = theano.function(inputs=[X, y], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=t)
        pass

    def floatX(self, X):
        return np.asarray(X, dtype = theano.config.floatX)

    def init_weights(self, shape, name):
        return theano.shared(self.floatX(np.random.randn(*shape) * 0.01), name=name)

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value()*0.)
            acc_new = rho * acc + (1-rho) * g**2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))

        return updates

    def feedforward(self, X, w1, w2, wo):
        a1 = T.nnet.sigmoid(T.dot(X, w1))
        a2 = T.nnet.sigmoid(T.dot(a1, w2))
        a_o = T.dot(a2, wo)
        return a_o

