import numpy as np
#import tensorflow as tf
import nn_eigval 
import matplotlib.pyplot as plt

#Same seed for all to debug
np.random.seed(42)

# Set all constants
matrixSize = 6
hidden_neurons = 50
num_layers = 4
max_epoch = 1000
learning_rate = 0.001
guess = 5
delta_treshold = 1e-17
verbose = True

M = nn_eigval.random_symmetric_matrix(matrixSize)
eigVals, eigVecs = np.linalg.eig(M)

TF_eigVal, TF_eigVecs, maxIterStep = nn_eigval.nn_eigval(
    M,
    matrixSize,
    hidden_neurons,
    num_layers,
    max_epoch,
    learning_rate,
    guess,
    delta_treshold,
    verbose
)

print(TF_eigVal, eigVals)

plt.figure()
t = np.linspace(0, maxIterStep+1, maxIterStep+1)
for i in range(matrixSize):
    plt.plot(t, TF_eigVecs[:, 0, i])

#Choose which eigenvector (find the correc tone by comparing values, run time is short)
eigVec = 0
for i in range(matrixSize):
    plt.axhline(y = eigVecs[0, i], ls = 'dashed', alpha = 0.7)


plt.show()





