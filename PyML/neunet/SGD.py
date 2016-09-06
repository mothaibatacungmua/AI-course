import numpy as np
import random
from monitor import monitor

def update_mini_batch(network, mini_batch, eta, lmbda, n):
    grad_b = [np.zeros(b.shape) for b in network.biases]
    grad_w = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        delta_grad_b, delta_grad_w = network.backprop(x, y)
        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

    network.weights = [ (1-eta*(lmbda/n))*w - (eta/len(mini_batch))*gw
                        for w,gw in zip(network.weights, grad_w)]

    network.biases = [b - (eta/len(mini_batch))*gb
                      for b, gb in zip(network.biases, grad_b)]


def SGD(network, training_data, epochs, mini_batch_size, eta,
        lmbda = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):

    n = len(training_data)

    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []

    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)
        ]

        print "epochs[%d]" % j

        for mini_batch in mini_batches:
            update_mini_batch(network, mini_batch, eta, lmbda, n)

        monitor(network, training_data, evaluation_data,
                training_cost,training_accuracy,evaluation_cost,evaluation_accuracy,
                lmbda,
                monitor_evaluation_cost, monitor_evaluation_accuracy,
                monitor_training_cost, monitor_training_accuracy)

    return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
