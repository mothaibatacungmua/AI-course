import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

#f(x) = exp(-(x-1)^2/(2x))*(x+1)/12
delta_x = 0.01
x = np.arange(0.01, 20, delta_x)
fig, ax = plt.subplots()

f = lambda x: np.exp(-(x-1)**2/(2*x)) * (x+1)/12
fx = f(x)
Fx = np.cumsum(fx) * delta_x

ax.plot(x, fx, label='$f(x)$')
ax.plot(x, Fx, label='$F(x)$', color='g')
ax.legend(loc=0, fontsize=16)
ax.set_title('True distribution')

M = .3  # scale factor
u1 = np.random.uniform(size=10000)*15
u2 = np.random.uniform(size=10000)
idx = np.where(u2 <= f(u1)/M)

samples = u1[idx]

fig, ax = plt.subplots()
ax.plot(x, fx, label='$f(x)$', linewidth=2)
sampling = ax.hist(samples, 30, normed=True, color='red')
ax.set_ylim([0.0, 1.0])
ax.set_title('Estimated distribution by uniform distribution')


fig, ax = plt.subplots()
ax.plot(u1, u2, '.', label='rejected', alpha=.3)
ax.plot(u1[idx], u2[idx], 'g.', label='accepted', alpha=.3)

ax.legend(fontsize=16)


ch = scipy.stats.chi2(4)
h = lambda x: f(x)/ch.pdf(x)

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].plot(x, fx, label='$f(x)$')
ax[0].plot(x, ch.pdf(x), label='$g(x)$')
ax[0].legend(loc=0, fontsize=18)
ax[1].plot(x, h(x))
ax[1].set_title('$h(x)=f(x)/g(x)$', fontsize=18)

hmax = h(x).max()
print hmax
u1 = ch.rvs(10000)
u2 = np.random.uniform(0.,1., 10000)
idx = np.where(u2 <= h(u1)/hmax)

v = u1[idx]
fig, ax = plt.subplots()
ax.hist(v, 30, normed=True)
ax.plot(x, fx, 'r', linewidth=2, label='$f(x)$')
ax.set_title('Estimated distribution by chi-square distribution')
ax.legend(fontsize=16)

fig, ax = plt.subplots()
ax.plot(u1, u2, '.', label='rejected', alpha=.3)
ax.plot(u1[idx], u2[idx], 'g.', label='accepted', alpha=.3)
ax.plot(x, h(x)/hmax, linewidth=2, label='$h(x)$')
ax.legend(fontsize=16)
plt.show()