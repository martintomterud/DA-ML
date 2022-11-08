# ----- PACKAGE IMPORT ----- #

#libraries
from math import degrees
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time 
import os
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#src
import franke
import dataFunctions
import prepareData
import sgd
#set seed for numpy random using time 
#np.random.seed(seed=int(time.time())) 
#set const seed for debugging
np.random.seed(seed=int(42)) 

# ---- END PACKAGE IMPORT ---- #


"""
This file contains functions that are executed to answer
parts a) of project 2

Function: sgd() compares sgd with and without momentum for given minibatch size
Function: adagrad answers a4
Function: moments answers a5
"""

def sgd(batchSize):

    # Set constants for use
    N = 500 #datapoints in grid
    degree = 6 #poly degree 
    mean = 0.0 #mean of franke
    std = 0.1  #std of franke

    # test parameters
    degrees = np.arange(1, 15)
    lambdas = np.logspace(-10, 0, 5)
    learningRates = np.logspace(-6, -1, 6)
    totEpochs = 1000
    epochsPerIteration  = 10
    numIterations = int(totEpochs/epochsPerIteration)
    numBatches = int(N / batchSize)
    momentum = 0.8
    
    #Storage container
    mse_Computed = np.zeros([len(learningRates), numIterations])
    mse_Computed_Momentum = np.zeros([len(learningRates), numIterations])

    #generate meshgrid and splitted franke data 
    #using the functions in prepareData.py
    #the data is scaled, centeresd and splitted
    x, y, fData, f_train, f_test, X, X_train, X_test = prepareData.initializeData(N, mean,std, degree, test_size = 0.2)


    # Initialize weights
    N_weights = int((degree + 1)*(degree + 2) / 2)
    initialWeights = np.random.randn(N_weights)

    #start loop using sgd only

    # compute intial val first and use it 
    # for subsequent runs
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.sgd(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            learningRate = learningRates[i], 
            costFunction = sgd.cost_OLS,
        )
        mse_Computed[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.sgd(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration, 
                learningRate = learningRates[i], 
                costFunction = sgd.cost_OLS,
            )
            mse_Computed[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)

    # now do one more loop using sgdm
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.sgdm(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            learningRate = learningRates[i],
            momentum = momentum, 
            costFunction = sgd.cost_OLS,
        )
        mse_Computed_Momentum[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.sgdm(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration, 
                learningRate = learningRates[i], 
                momentum = momentum, 
                costFunction = sgd.cost_OLS,
            )
            mse_Computed_Momentum[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        
        plt.subplots(1, 2)
        plt.subplot(121)

        #need to make functions
