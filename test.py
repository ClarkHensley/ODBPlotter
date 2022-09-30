import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes()
fig.add_axes(ax)

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

ax.imshow(gradient, cmap="turbo")

plt.show()
