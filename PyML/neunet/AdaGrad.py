import numpy as np
import random
from monitor import monitor

'''
AdaGrad's intuition explaining
https://www.youtube.com/watch?v=0qUAb94CpOw
'''


def update_mini_batch(
        network, mini_batch, eta, lmbda, n,
        epsilon, AdaGrad_b, AdaGrad_w):
    grad_b = [np.zeros(b.shape) for b in network.biases]
    grad_w = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        delta_grad_b, delta_grad_w = network.backprop(x, y)
        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

    AdaGrad_b = [past + (gb/len(mini_batch)) ** 2
                 for past, gb in zip(AdaGrad_b, grad_b)]
    AdaGrad_w = [past + (gw/len(mini_batch) + (lmbda/n)*w) ** 2
                 for past, gw, w in zip(AdaGrad_w, grad_w,network.weights)]

    network.weights = [w - (eta/np.sqrt(ada_w + epsilon))*(gw/len(mini_batch) + (lmbda/n)*w)
                        for w, gw, ada_w in zip(network.weights, grad_w, AdaGrad_w)]

    network.biases = [b - (eta/np.sqrt(ada_b + epsilon))*(gb/len(mini_batch))
                      for b, gb, ada_b in zip(network.biases, grad_b, AdaGrad_b)]

    return AdaGrad_b, AdaGrad_w


def AdaGrad(
        network, training_data, epochs, mini_batch_size, eta,
        epsilon=0.00000001, lmbda = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):

    n = len(training_data)
    AdaGrad_b = [np.zeros(b.shape) for b in network.biases]
    AdaGrad_w = [np.zeros(w.shape) for w in network.weights]

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
            AdaGrad_b, AdaGrad_w = update_mini_batch(
                                        network, mini_batch, eta, lmbda, n,
                                        epsilon, AdaGrad_b, AdaGrad_w
                                    )

        monitor(network, training_data, evaluation_data,
                training_cost,training_accuracy,evaluation_cost,evaluation_accuracy,
                lmbda,
                monitor_evaluation_cost, monitor_evaluation_accuracy,
                monitor_training_cost, monitor_training_accuracy)

    return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
