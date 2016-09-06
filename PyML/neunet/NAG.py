import numpy as np
import random
from monitor import monitor


def update_mini_batch(
        network, mini_batch,
        eta, momentum, lmbda, n,
        veloc_b, veloc_w):

    biases = [b - momentum * curr_v for (b, curr_v) in zip(network.biases, veloc_b)]
    weights = [w - momentum * curr_v for (w, curr_v) in zip(network.weights, veloc_w)]
    grad_b = [np.zeros(b.shape) for b in network.biases]
    grad_w = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        #compute gradient descents
        delta_grad_b, delta_grad_w = network.backprop(x, y, biases=biases, weights=weights)
        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

    # update bias, weight velocities
    veloc_b = [momentum * v + (eta/len(mini_batch)) * gb for (v, gb) in zip(veloc_b, grad_b)]
    veloc_w = [momentum * v + (eta/len(mini_batch)) * gw for (v, gw) in zip(veloc_w, grad_w)]

    #update bias, weight with velocities
    network.biases = [b - v for (b,v) in zip(network.biases, veloc_b)]
    network.weights = [(1-eta*(lmbda/n))*w - v for (w, v) in zip(network.weights, veloc_w)]

    return veloc_b, veloc_w


def NAG(network, training_data, epochs, mini_batch_size, eta,
       momentum=0.9, lmbda=0.0,
       evaluation_data=None,
       monitor_evaluation_cost=False,
       monitor_evaluation_accuracy=False,
       monitor_training_cost=False,
       monitor_training_accuracy=False):
    n = len(training_data)

    veloc_b = [np.zeros(b.shape) for b in network.biases]
    veloc_w = [np.zeros(w.shape) for w in network.weights]

    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []

    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in xrange(0, n, mini_batch_size)
            ]

        print "epochs[%d]" % j

        for mini_batch in mini_batches:
            veloc_b, veloc_w = update_mini_batch(
                network, mini_batch, eta,
                momentum, lmbda, n,
                veloc_b, veloc_w)

        monitor(network, training_data, evaluation_data,
                training_cost, training_accuracy, evaluation_cost, evaluation_accuracy,
                lmbda,
                monitor_evaluation_cost, monitor_evaluation_accuracy,
                monitor_training_cost, monitor_training_accuracy)

    return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
    pass
