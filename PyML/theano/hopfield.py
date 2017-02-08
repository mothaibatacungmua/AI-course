import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import random

N = 4
X = T.dmatrix('X')
dg = np.ones(N, dtype=theano.config.floatX)
W = theano.shared(value=np.zeros((N,N), dtype=theano.config.floatX))

f = T.dot(T.transpose(X), X) - T.shape(X)[0]*T.nlinalg.alloc_diag(dg)

input = np.asarray([[1,-1,1,-1],[-1,-1,1,-1],[1,1,1,1],[1,1,-1,-1],[-1,-1,1,1]], dtype=theano.config.floatX)

train = theano.function(inputs=[X],outputs=f,updates=[(W, W+f)])

train(input)
print W.get_value()

x = T.dvector('x')
energy = -1/2*T.dot(T.dot(T.transpose(x), W),x)

e_f0 = theano.function(inputs=[x], outputs=energy)

#reconstruction

state = theano.shared(value=np.zeros((N,), dtype=theano.config.floatX))

srng = RandomStreams(seed=234)
idx = T.lscalar()
ret = T.dot(W[idx,:], state)
new_state = ifelse(T.le(ret,np.cast[theano.config.floatX](0)),-1.0,1.0)
updates = [(state, T.set_subtensor(state[idx], new_state))]

e_f1 = theano.function(inputs=[x, idx], outputs=energy, updates=updates)

state.set_value([1,1,1,-1])
pre_e = 0.0
next_e = e_f0(state.get_value())

for i in range(0, 20):
    print "State %s with energy %d" % (str(state.get_value()), next_e)
    pre_e = next_e
    e_f1(state.get_value(), random.randint(0, N-1))
    next_e = e_f0(state.get_value())

# 2 local minima energy
#  State [ 1.  1. -1. -1.] with energy -16
#  State [-1. -1.  1.  1.] with energy -16