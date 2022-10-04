"""
This file is created for testing code functionality
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import franke
import dataFunctions
import designMatrix
import regressionClass

# Set up grid
N = 100
x, y = dataFunctions.meshgrid(N)

normal_franke = franke.franke(x, y)
noisy_franke = franke.noisy_franke(x, y, 0.0, 0.2, N)


fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface 
surf = ax.plot_surface(x, y, noisy_franke, cmap=cm.winter,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

deg = 4
X = designMatrix.designMatrix(N, deg)
X.designMatrix(x, y, deg)


