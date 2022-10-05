import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
import os

"""
This code is taken from MHJs GitHub
Modified wih correct path to data
"""

# Load the terrain
cd = os.path.abspath('.')
datapath = cd + "/data/"
terrain = imread(datapath + 'SRTM_data_Norway_1.tif')
print(terrain.size)
N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]
print(terrain.size)
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain
#X = create_X(x_mesh, y_mesh,m)


# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()