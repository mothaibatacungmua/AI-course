import numpy as np


class SoftmaxCost(object):
    @staticmethod
    def fn(a, y):
        return -np.nan_to_num(np.log(a[np.argmax(y)]))

    @staticmethod
    def delta(z, a, y):
        return (a - y)

    @staticmethod
    def output_activation(a, w, b):
        z = np.dot(w, a) + b
        outputs = np.exp(z)
        outputs = outputs/np.sum(outputs)
        return outputs