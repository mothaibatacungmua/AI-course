import random
import numpy as np
from libs import sigmoid, sigmoid_prime

"""
~~~~~~~~
NOTE: A bag of tricks for mini-batch gradient descent
1: Initializing the weights randomly and rescaling the weights by sqrt
2: Shifting the inputs to change the error surface, making the means of input is zero
3: Scaling the inputs to change the error surface, making the variances of input is one
"""

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[:1])]

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        pass

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                return (self.evaluate(test_data), n_test)

        pass

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]

        return sum(int(x == y) for (x,y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [gd + dgd for gd, dgd in zip(grad_b, delta_grad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        self.weights = [w - (eta/len(mini_batch))*gdw for w, gdw in zip(self.weights, grad_w)]
        self.biases = [w - (eta/len(mini_batch))*gdb for b, gdb in zip(self.biases, grad_b)]
        pass

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        #foward pass
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].tranpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].tranpose(), delta)*sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].tranpose())

        return (grad_b, grad_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

