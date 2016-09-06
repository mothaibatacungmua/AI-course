import numpy as np
import random
from monitor import monitor


def update_mini_batch(
        network, mini_batch, eta,
        lmbda, n, epsilon, time,
        fraction_1, fraction_2,
        moments_w, moments_b):
    grad_b = [np.zeros(b.shape) for b in network.biases]
    grad_w = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        delta_grad_b, delta_grad_w = network.backprop(x, y)
        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

    moments_b = [(fraction_1*past[0]+(1-fraction_1)*(gb/len(mini_batch)),
                  fraction_2*past[1] + (1-fraction_2)*(gb/len(mini_batch)) ** 2)
                 for past, gb in zip(moments_b, grad_b)]

    moments_w = [(fraction_1*past[0]+(1-fraction_1)*(gw/len(mini_batch) + (lmbda/n)*w),
                  fraction_2*past[1] + (1-fraction_2)*(gw/len(mini_batch) + (lmbda/n)*w) ** 2)
                 for past, gw, w in zip(moments_w, grad_w,network.weights)]

    delta_b = [eta*m[0]/(1-fraction_1**time)/(np.sqrt(m[1]/(1-fraction_2**time)) + epsilon)
               for m in moments_b]
    delta_w = [eta*m[0]/(1-fraction_1**time)/(np.sqrt(m[1]/(1-fraction_2**time)) + epsilon)
               for m in moments_w]

    network.weights = [w - delta for w, delta in zip(network.weights, delta_w)]

    network.biases = [b - delta for b, delta in zip(network.biases, delta_b)]

    return moments_b, moments_w


def Adam(
        network, training_data, epochs, mini_batch_size, eta,
        epsilon=0.00000001, lmbda = 0.0,
        fraction_1=0.9,fraction_2=0.9999,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):

    n = len(training_data)
    moments_b = [(np.zeros(b.shape),np.zeros(b.shape)) for b in network.biases]
    moments_w = [(np.zeros(w.shape),np.zeros(w.shape)) for w in network.weights]

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
            moments_b, moments_w = update_mini_batch(
                                        network, mini_batch, eta, lmbda, n,
                                        epsilon, j+1, fraction_1, fraction_2,
                                        moments_b, moments_w
                                    )

        monitor(network, training_data, evaluation_data,
                training_cost,training_accuracy,evaluation_cost,evaluation_accuracy,
                lmbda,
                monitor_evaluation_cost, monitor_evaluation_accuracy,
                monitor_training_cost, monitor_training_accuracy)

    return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
