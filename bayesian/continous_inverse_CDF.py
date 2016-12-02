import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

alpha = 1.
nsamp = 10000

#f = alpha * exp(-alpha * x)
#F = 1 - exp(-alpha * x)
x = np.arange(0, 10, 0.01)
f = alpha* np.exp(-alpha * x)
fig, ax = plt.subplots()
real, = ax.plot(x, f, linewidth=2)
ax.text(1.5, 0.9, r'$f(x)=1 - e^{-\alpha x}$', fontsize=14)
ax.grid(True)
ax.set_title('Inverse PDF for Exponential Distribution')

u = np.random.uniform(0, 1, 10000)
samples = 1/alpha*np.log(1/(1-u)) #note: shifting 1 for CDF

sampling = ax.hist(samples, 30, normed=True, color='red')

red_patch = mpatches.Patch(color='red')
plt.legend([real, red_patch], ['Real', 'Estimasted'])
plt.show()