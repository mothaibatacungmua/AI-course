import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 6, 0.01)
y = [np.floor(x[i])/6 for i in range(0, len(x))]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid(True)
ax.set_title('CDF of Fair Six-Sided Side')


u = np.random.uniform(0, 1, 10000)
dice_samples = []
for i in u:
    if 0 <= i and i < 1.0/6:
        dice_samples.append(1)
    elif 1.0/6 <= i and i < 2.0/6:
        dice_samples.append(2)
    elif 2.0/6 <= i and i < 3.0/6:
        dice_samples.append(3)
    elif 3.0 / 6 <= i and i < 4.0 / 6:
        dice_samples.append(4)
    elif 4.0 / 6 <= i and i < 5.0 / 6:
        dice_samples.append(5)
    elif 5.0 / 6 <= i and i < 6.0 / 6:
        dice_samples.append(6)

probs = [0.]
for i in range(1,7):
    probs.append(dice_samples.count(i))

probs = np.array(probs)*1. / len(dice_samples)
estimated_cdf = np.cumsum(probs)
outcomes = np.arange(0, 7, 1)

print probs
fig, ax = plt.subplots()
ax.plot(outcomes, estimated_cdf)
ax.grid(True)
ax.set_title('Estimated CDF of Fair Six-Sided Side')

plt.show()