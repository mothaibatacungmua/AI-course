import matplotlib.pyplot as plt
import numpy as np


def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs),
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')

    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()


def plot_tranining_accuracy(training_accuracy, num_epochs, tranining_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(tranining_accuracy_xmin, num_epochs),
            training_accuracy[tranining_accuracy_xmin:num_epochs],
            color='#2A6EA6')

    ax.set_xlim([tranining_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()


def plot_test_cost(test_cost, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs),
            test_cost[test_accuracy_xmin:num_epochs],
            color='#2A6EA6')

    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()


def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs),
            test_accuracy[test_accuracy_xmin:num_epochs],
            color='#2A6EA6')

    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()


def plot_comparing(training_cost, test_cost, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot tranining costs
    ax.plot(np.arange(xmin, num_epochs),
            training_cost[xmin:num_epochs],
            label='Tranining cost',
            color='#2A6EA6')

    # Plot test costs
    ax.plot(np.arange(xmin, num_epochs),
            test_cost[xmin:num_epochs],
            label='Test cost',
            color='#FFCD33')

    ax.set_xlim([xmin, num_epochs-1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()