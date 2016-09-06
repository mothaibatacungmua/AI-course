from advance_network import Network
from libs import neunet_data_wrapper
from quadratic_cost import QuadraticCost
from cross_entropy_cost import CrossEntropyCost
from softmax_cost import SoftmaxCost
from libs import plot_comparing, plot_training_cost, plot_test_cost

training_data, validation_data, test_data = neunet_data_wrapper()

print 'Trainning...'
quadratic_mnist_net = Network([784, 30, 10], cost=QuadraticCost)
cross_entropy_mnist_net = Network([784, 30, 10], cost=CrossEntropyCost)
softmax_mnist_net = Network([784, 30, 10], cost=SoftmaxCost)

epochs = 30
training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = \
    cross_entropy_mnist_net.AdaGrad(training_data, epochs, 10, 0.001,
                        lmbda=0.1, epsilon=0.00000001, evaluation_data=test_data,
                        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
                        monitor_training_cost=True, monitor_training_accuracy=True)

#plot_training_cost(training_cost, epochs, 0)
#plot_test_cost(evaluation_cost, epochs, 0)
#print evaluation_accuracy
plot_comparing(training_cost, evaluation_cost, epochs, 0)

print 'Test accuracy: %0.2f' % evaluation_accuracy[-1]
print 'Finished!'