import numpy as np
from libs import sigmoid_prime, sigmoid


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)

    @staticmethod
    def output_activation(a, w, b):
        return sigmoid(np.dot(w,a)+b)