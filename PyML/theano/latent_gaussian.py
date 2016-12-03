import matplotlib.pyplot as plt
import numpy as np
import theano
from theano import tensor as T

epsilon = np.array(1e-4, dtype=theano.config.floatX)
def normal_cdf(x, location=0, scale=1):
    location = T.cast(location, theano.config.floatX)
    scale = T.cast(scale, theano.config.floatX)

    div = T.sqrt(2 * scale ** 2 + epsilon)
    div = T.cast(div, theano.config.floatX)

    erf_arg = (x - location) / div
    return .5 * (1 + T.erf(erf_arg + epsilon))

rng = np.random.RandomState(123456)

N = 500
hidden_nodes = 30
e_samples = np.asarray(np.random.exponential(0.5, N), dtype=theano.config.floatX)
log_samples = np.log(e_samples)
fig, ax = plt.subplots()
ax.hist(e_samples, 30, normed=True, color='red')
ax.set_title('Origin Skewness Distribution')

Wh = theano.shared(value=np.asarray(rng.uniform(-1, 1, (hidden_nodes,)), dtype=theano.config.floatX), borrow=True)
bh = theano.shared(value=np.zeros((hidden_nodes,), dtype=theano.config.floatX),borrow=True)
Wo = theano.shared(value=np.asarray(rng.normal(size=(hidden_nodes,)), dtype=theano.config.floatX),borrow=True)
bo = theano.shared(np.cast[theano.config.floatX](0.0),borrow=True)

params = [Wh, bh, Wo, bo]

x = T.dvector('x')
shapex = T.shape(x)[0]
h = T.outer(Wh,x) + T.transpose(T.tile(bh,(shapex,1)))
f = theano.function([x], h)
print f(e_samples)

a = 1/(1 + T.exp(-h))

output = T.dot(Wo,a) + T.tile(bo, (1,shapex))
sorted_output = output.sort()
pe = T.dvector('x')

output_fn = theano.function([x], output)
#fff = theano.function([x], sorted_output)
cost = T.sqr(normal_cdf(sorted_output) - pe).sum()/shapex
cost_fn = theano.function([x, pe], cost)

uniform_pe = np.asarray(np.arange(1, N+1), dtype=theano.config.floatX)/N

print cost_fn(e_samples, uniform_pe)

#grad = [T.grad(cost, param) for param in params]
learning_rate = 0.25
gamma = 0.9


# momentum
updates = []
for param in params:
    param_update = theano.shared(param.get_value() * 0.)
    updates.append((param, param - learning_rate * param_update))
    updates.append((param_update, gamma * param_update + (1. - gamma) * T.grad(cost, param)))


train = theano.function(inputs=[x, pe],outputs=cost,updates=updates)
#theano.scan(lambda ,sequences=output)
#plt.show()
nloops = 40000
for i in range(0, nloops):
    train_cost = train(log_samples, uniform_pe)
    if (i% 1000) == 0:
        print "iter:%d, cost:%08f\n" % (i, train_cost)


latent_gaussian = output_fn(log_samples)
latent_gaussian = np.reshape(latent_gaussian, (N, 1))

fig, ax = plt.subplots()
ax.hist(latent_gaussian, 30, normed=True, color='red')
ax.set_title('Latent variable Gaussian distribution')

plt.show()