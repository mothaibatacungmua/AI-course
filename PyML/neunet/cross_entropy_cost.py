import numpy as np
from libs import sigmoid, sigmoid_prime

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)

    @staticmethod
    def output_activation(a, w, b):
        return sigmoid(np.dot(w, a) + b)