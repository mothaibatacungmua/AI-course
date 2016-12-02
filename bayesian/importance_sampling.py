import numpy as np
import scipy.stats

# Monte Carlo
N = 100

mc = []
h = lambda x: (x >= 3) * 1.0
for i in range(0, 500):
    samples = np.random.normal(0.,1.,N)
    mc.append(np.sum(h(samples))/N)

mc_mean = np.sum(mc)/500
mc_var = np.var(mc)

print 'Monte Carlo method, mean:%07f, var:%07f' % (mc_mean, mc_var)
# Importance sampling
isp_unbiased = []
isp_biased = []

f = scipy.stats.norm(0., 1.).pdf
g = scipy.stats.norm(4., 1.).pdf


for i in range(0, 200):
    samples = np.random.normal(4., 1., N)
    isp_unbiased.append(np.sum(f(samples)*h(samples)/g(samples))/N)

g = scipy.stats.norm(2., 1.).pdf
for i in range(0, 200):
    samples = np.random.normal(2., 1., N)
    isp_biased.append(np.sum(f(samples)*h(samples)/g(samples))/np.sum(f(samples)/g(samples)))

print 'Unbiased Importance Sampling method, mean:%07f, var:%07f' % (np.sum(isp_unbiased)/200, np.var(isp_unbiased))
print 'Biased Importance Sampling method, mean:%07f, var:%07f' % (np.sum(isp_biased)/200, np.var(isp_biased))