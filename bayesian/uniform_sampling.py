import numpy as np
import matplotlib.pyplot as plt

s = np.random.uniform(0, 1, 10000)
count, bins, ignored = plt.hist(s, 20, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

plt.show()
