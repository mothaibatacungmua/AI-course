import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import random

Nh = 5
Nv = 7

learning_rate = 0.1