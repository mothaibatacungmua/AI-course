import random
import numpy as np
from cross_entropy_cost import CrossEntropyCost
from libs import sigmoid, sigmoid_prime
from libs import vectorized_result
from SGD import SGD
from CM import CM
from NAG import NAG
from AdaGrad import AdaGrad
from Adadelta import Adadelta
from RMSprop import RMSprop
from Adam import Adam


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost, init_method='default'):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = None
        self.weights = None

        if init_method == 'default':
            self.default_weight_initializer()

        if init_method == 'large':
            self.large_weight_initialize()

        self.cost = cost
        pass

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        pass

    def large_weight_initialize(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        pass

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a
        pass

    def backprop(self, x, y, biases=None, weights=None):
        if biases is None: biases = self.biases
        if weights is None: weights = self.weights

        grad_b = [np.zeros(b.shape) for b in biases]
        grad_w = [np.zeros(w.shape) for w in weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #computing the activation of the output layer
        activations[-1] = self.cost.output_activation(activations[-2], weights[-1], biases[-1])

        #backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (grad_b, grad_w)
        pass

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)

        cost += 0.5*(lmbda/len(data)) * sum((np.linalg.norm(w)**2 for w in self.weights))

        return cost

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]

        return sum(int(t == y) for (t, y) in results)

    def SGD(self, *args, **kwargs):
        return SGD(self, *args, **kwargs)

    def CM(self, *args, **kwargs):
        return CM(self, *args, **kwargs)

    def NAG(self, *args, **kwargs):
        return NAG(self, *args, **kwargs)

    def AdaGrad(self, *args, **kwargs):
        return AdaGrad(self, *args, **kwargs)

    def Adadelta(self, *args, **kwargs):
        return Adadelta(self, *args, **kwargs)

    def L_BFGS(self, *args, **kwargs):
        pass

    def RMSprop(self, *args, **kwargs):
        return RMSprop(self, *args, **kwargs)

    def Adam(self, *args, **kwargs):
        return Adam(self, *args, **kwargs)