import numpy as np
import random
from monitor import monitor


def update_mini_batch(
        network, mini_batch, eta, lmbda, n,
        epsilon, fraction, RMS_b, RMS_w):
    grad_b = [np.zeros(b.shape) for b in network.biases]
    grad_w = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        delta_grad_b, delta_grad_w = network.backprop(x, y)
        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

    RMS_b = [fraction*past + (1-fraction)*(gb/len(mini_batch)) ** 2
             for past, gb in zip(RMS_b, grad_b)]
    RMS_w = [fraction*past + (1-fraction)*(gw/len(mini_batch) + (lmbda/n)*w) ** 2
             for past, gw, w in zip(RMS_w, grad_w,network.weights)]

    network.weights = [w - (eta/np.sqrt(rms_w + epsilon))*(gw/len(mini_batch) + (lmbda/n)*w)
                        for w, gw, rms_w in zip(network.weights, grad_w, RMS_w)]

    network.biases = [b - (eta/np.sqrt(ada_b + epsilon))*(gb/len(mini_batch))
                      for b, gb, ada_b in zip(network.biases, grad_b, RMS_b)]

    return RMS_b, RMS_w


def RMSprop(
        network, training_data, epochs, mini_batch_size, eta,
        epsilon=0.00000001, lmbda = 0.0, fraction=0.9,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):

    n = len(training_data)
    RMS_b = [np.zeros(b.shape) for b in network.biases]
    RMS_w = [np.zeros(w.shape) for w in network.weights]

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
            RMS_b, RMS_w = update_mini_batch(
                                        network, mini_batch, eta, lmbda, n,
                                        epsilon, fraction, RMS_b, RMS_w
                                    )

        monitor(network, training_data, evaluation_data,
                training_cost,training_accuracy,evaluation_cost,evaluation_accuracy,
                lmbda,
                monitor_evaluation_cost, monitor_evaluation_accuracy,
                monitor_training_cost, monitor_training_accuracy)

    return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
