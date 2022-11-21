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
#scikit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression #for comparisson

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

Function: sgd_main() compares sgd with and without momentum for given minibatch size
Function: sgd_ridge() compares the two best learning rates of sgd_main with differnet
            l2 penalties added
Function: adaptiveGradient() looks at the performance of the adagrad method
Function: tunableRates() compares the ADAM and RMSprop methods for tuning the learnign rates

The function calls as they stand produce the figures in the report.
"""

def sgd_main(batchSize):
    # Set constants for use
    N = 1000 #datapoints in grid
    degree = 5 #poly degree 
    noiseScale = 0.1
    mean = 0.0 #mean of franke
    std = 0.1  #std of franke

    # test parameters
    learningRates = np.logspace(-6, -2, 5)
    totEpochs = 2500
    epochsPerIteration  = 10
    numIterations = int(totEpochs/epochsPerIteration)
    numBatches = int(N / batchSize)
    momentum = 0.6
    
    #Storage container
    mse_Computed = np.zeros([len(learningRates), numIterations])
    mse_Computed_Momentum = np.zeros([len(learningRates), numIterations])
    
    X_train, X_test, f_train, f_test = prepareData.prepForSGD_Scale(N, noiseScale, degree)
    
    #compute sckikit learn lin reg to compare with ols

    reg = LinearRegression().fit(X_train, f_train)
    reg_coeffs = reg.coef_
    mse_sklearn = dataFunctions.mse(f_test, X_test @ reg_coeffs)

    # Initialize weights
    N_weights = int((degree + 1)*(degree + 2) / 2 - 1) 
    initialWeights = np.random.randn(N_weights)

    #start loop using sgd only
    # compute intial val first and use it 
    # for subsequent runs
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.SGD(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            learningRate = learningRates[i], 
            costFunction = sgd.cost_OLS_derivative,
        )
        mse_Computed[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
    
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.SGD(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration, 
                learningRate = learningRates[i], 
                costFunction = sgd.cost_OLS_derivative,
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
            costFunction = sgd.cost_OLS_derivative,
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
                costFunction = sgd.cost_OLS_derivative,
            )
            mse_Computed_Momentum[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)

    fig, ax  = plt.subplots(1, 2)
    fig.suptitle('Minibatch size = ' + str(batchSize), fontsize = 14)
    fig.tight_layout()
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax[0].xaxis.set_major_locator(MultipleLocator(1000))
    ax[1].xaxis.set_major_locator(MultipleLocator(1000))
    ax[0].xaxis.set_minor_locator(MultipleLocator(500))
    ax[1].xaxis.set_minor_locator(MultipleLocator(500))
    #ax[0].yaxis.set_major_locator(MultipleLocator(0.01))
    #ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for a in ax:
        a.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        a.tick_params(axis='x', top=True, direction = 'in', length = 10)
        a.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        a.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('Number of epochs')
    ax[1].set_xlabel('Number of epochs')
    ax[0].set_title('SGD', fontsize = 14)
    ax[1].set_title(r'SGDM, $\eta = $' + str(momentum), fontsize = 14)

    x_axis = np.linspace(1, totEpochs, numIterations)
    colors = ['r', 'b', 'g', 'darkorange', 'hotpink', 'teal', 'navy']
    for i in range(len(learningRates)):
        ax[0].semilogy(x_axis, mse_Computed[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
        ax[1].plot(x_axis, mse_Computed_Momentum[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
    for a in ax:
        a.axhline(y = mse_sklearn, color = 'k', label = 'sklearn OLS')
    ax[0].legend(fontsize = 12)
    plt.savefig('sgd_1.pdf', bbox_inches = 'tight')
    # plt.show()

def sgd_ridge(batchSize):
    """
    Function that explores different lambda-parameters for the best learning rates found with
    sgd on ols using the ridge cost function
    """
    # Set constants for use
    N = 1000 #datapoints in grid
    degree = 5 #poly degree 
    noiseScale = 0.1

    # test parameters
    learningRates = np.logspace(-3, -2, 2)
    lambdas = np.logspace(-6, 0, 7)
    totEpochs = 2500
    epochsPerIteration  = 10
    numIterations = int(totEpochs/epochsPerIteration)
    numBatches = int(N / batchSize)
    momentum = 0.6
    
    #Storage container
    mse_Computed = np.zeros([len(learningRates), len(lambdas) ,numIterations])
    mse_Computed_Momentum = np.zeros([len(learningRates), len(lambdas), numIterations])
    
    X_train, X_test, f_train, f_test = prepareData.prepForSGD_Scale(N, noiseScale, degree)
    
    reg = LinearRegression().fit(X_train, f_train)
    reg_coeffs = reg.coef_
    mse_sklearn = dataFunctions.mse(f_test, X_test @ reg_coeffs)

    # Initialize weights
    N_weights = int((degree + 1)*(degree + 2) / 2 - 1) 
    initialWeights = np.random.randn(N_weights)
    print(X_test.shape, X_train.shape, initialWeights.shape)
    #start loop using sgd only
    # compute intial val first and use it 
    # for subsequent runs
    for k in range(len(learningRates)):
        for i in range(len(lambdas)):
            weights_OLS_SGD = sgd.sgdm(
                X_train, 
                f_train, 
                initialWeights, 
                numBatches, 
                epochsPerIteration, 
                learningRates[k], 
                momentum,
                sgd.cost_Ridge_derivative,
                lambdas[i] 
            )
            mse_Computed[k, i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        
            #loop over iterations
            for j in range(1, numIterations):
                weights_OLS_SGD = sgd.sgdm(
                    X_train, 
                    f_train, 
                    weights_OLS_SGD, 
                    numBatches, 
                    epochsPerIteration, 
                    learningRates[k], 
                    momentum,
                    sgd.cost_Ridge_derivative,
                    lambdas[i] 
                )
                mse_Computed[k, i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
    fig, ax  = plt.subplots(1, 2)
    fig.tight_layout()
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax[0].xaxis.set_major_locator(MultipleLocator(1000))
    ax[1].xaxis.set_major_locator(MultipleLocator(1000))
    ax[0].xaxis.set_minor_locator(MultipleLocator(500))
    ax[1].xaxis.set_minor_locator(MultipleLocator(500))
    #ax[0].yaxis.set_major_locator(MultipleLocator(0.01))
    #ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for a in ax:
        a.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        a.tick_params(axis='x', top=True, direction = 'in', length = 10)
        a.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        a.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('Number of epochs')
    ax[1].set_xlabel('Number of epochs')
    ax[0].set_title(r'$\gamma = $' + str(learningRates[0]), fontsize = 14)
    ax[1].set_title(r'$\gamma = $' + str(learningRates[1]), fontsize = 14)

    x_axis = np.linspace(1, totEpochs, numIterations)
    colors = ['r', 'b', 'g', 'darkorange', 'hotpink', 'teal', 'navy', 'grey']
    for i in range(len(lambdas)):
        ax[0].plot(x_axis, mse_Computed[0, i, :], color = colors[i], label = r'$\lambda = 10^{%i}$' %int(np.log10(lambdas[i]))) 
        ax[1].plot(x_axis, mse_Computed[1, i, :], color = colors[i], label = r'$\lambda = 10^{%i}$' %int(np.log10(lambdas[i]))) 
    for a in ax:
        a.axhline(y = mse_sklearn, color = 'k', label = 'sklearn OLS')
    ax[0].legend(fontsize = 12)
    #plt.savefig('sgd_ridge_1.pdf', bbox_inches = 'tight')
    plt.show()

def adaptiveGradient(batchSize):
    """
    Function that tests adaptive gradient method
    compares run with momentum = 0
    and momentum finite
    """
    # Set constants for use
    N = 1000 #datapoints in grid
    degree = 5 #poly degree 
    noiseScale = 0.1

    # test parameters
    learningRates = np.logspace(-6, -2, 5)
    totEpochs = 2500
    epochsPerIteration  = 10
    numIterations = int(totEpochs/epochsPerIteration)
    numBatches = int(N / batchSize)
    momentum = 0.6
    
    #Storage container
    mse_Computed = np.zeros([len(learningRates), numIterations])
    mse_Computed_Momentum = np.zeros([len(learningRates), numIterations])
    
    X_train, X_test, f_train, f_test = prepareData.prepForSGD_Scale(N, noiseScale, degree)

    reg = LinearRegression().fit(X_train, f_train)
    reg_coeffs = reg.coef_
    mse_sklearn = dataFunctions.mse(f_test, X_test @ reg_coeffs)

    # Initialize weights
    N_weights = int((degree + 1)*(degree + 2) / 2 - 1) 
    initialWeights = np.random.randn(N_weights)

    #start loop using sgd only
    # compute intial val first and use it 
    # for subsequent runs
    # run with momentum = 0
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.adaptiveGradient(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            momentum= 0,
            learningRate = learningRates[i], 
            costFunction = sgd.cost_OLS_derivative,
        )
        mse_Computed[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
    
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.adaptiveGradient(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration, 
                momentum= 0,
                learningRate = learningRates[i], 
                costFunction = sgd.cost_OLS_derivative,
            )
            mse_Computed[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
   

    # now do one more loop using finite momentum
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.adaptiveGradient(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            momentum = momentum,
            learningRate = learningRates[i], 
            costFunction = sgd.cost_OLS_derivative,
        )
        mse_Computed_Momentum[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.adaptiveGradient(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration,
                momentum = momentum, 
                learningRate = learningRates[i], 
                costFunction = sgd.cost_OLS_derivative,
            )
            mse_Computed_Momentum[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)

    fig, ax  = plt.subplots(1, 2, sharey=True)
    plt.gca().set_ylim(bottom=0.014, top = 2.3)
    fig.suptitle('Minibatch size = ' + str(batchSize), fontsize = 14)
    fig.tight_layout()
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax[0].xaxis.set_major_locator(MultipleLocator(1000))
    ax[1].xaxis.set_major_locator(MultipleLocator(1000))
    ax[0].xaxis.set_minor_locator(MultipleLocator(500))
    ax[1].xaxis.set_minor_locator(MultipleLocator(500))
    #ax[0].yaxis.set_major_locator(MultipleLocator(0.01))
    #ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for a in ax:
        a.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        a.tick_params(axis='x', top=True, direction = 'in', length = 10)
        a.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        a.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('Number of epochs')
    ax[1].set_xlabel('Number of epochs')
    ax[0].set_title(r'$\eta = 0$', fontsize = 14)
    ax[1].set_title(r'$\eta = $' + str(momentum), fontsize = 14)

    x_axis = np.linspace(1, totEpochs, numIterations)
    colors = ['r', 'b', 'g', 'darkorange', 'hotpink', 'teal', 'navy']
    for i in range(len(learningRates)):
        ax[0].semilogy(x_axis, mse_Computed[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
        ax[1].semilogy(x_axis, mse_Computed_Momentum[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
    for a in ax:
        a.axhline(y = mse_sklearn, color = 'k', label = 'sklearn OLS')
    ax[0].legend(fontsize = 12, loc = 7)
    plt.savefig('adagrad_1.pdf', bbox_inches = 'tight')
    plt.show()

def tuneableRates(batchSize):
    """
    Function that tests compares 
    ADAM and RMSprop
    """
    # Set constants for use
    N = 1000 #datapoints in grid
    degree = 5 #poly degree 
    noiseScale = 0.1

    # test parameters
    learningRates = np.logspace(-6, -2, 5)
    totEpochs = 2500
    epochsPerIteration  = 1
    numIterations = int(totEpochs/epochsPerIteration)
    numBatches = int(N / batchSize)
    
    #Storage container
    mse_adam = np.zeros([len(learningRates), numIterations])
    mse_rms= np.zeros([len(learningRates), numIterations])
    
    X_train, X_test, f_train, f_test = prepareData.prepForSGD_Scale(N, noiseScale, degree)

    reg = LinearRegression().fit(X_train, f_train)
    reg_coeffs = reg.coef_
    mse_sklearn = dataFunctions.mse(f_test, X_test @ reg_coeffs)

    # Initialize weights
    N_weights = int((degree + 1)*(degree + 2) / 2 - 1) 
    initialWeights = np.random.randn(N_weights)

    #loop for rms
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.RMSporp(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            learningRate = learningRates[i], 
            discountFactor= 0.99,
            costFunction = sgd.cost_OLS_derivative,
        )
        mse_rms[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
    
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.RMSporp(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration, 
                learningRate = learningRates[i], 
                discountFactor= 0.99,
                costFunction = sgd.cost_OLS_derivative,
            )
            mse_rms[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
   

    # loop for adam
    for i in range(len(learningRates)):
        weights_OLS_SGD = sgd.ADAM(
            x = X_train, 
            y = f_train, 
            startWeights = initialWeights, 
            numBatches = numBatches, 
            numEpochs = epochsPerIteration, 
            learningRate = learningRates[i], 
            firstMomentFactor= 0.9,
            secondMomentFactor= 0.999,
            costFunction = sgd.cost_OLS_derivative,
        )
        mse_adam[i, 0] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)
        #loop over iterations
        for j in range(1, numIterations):
            weights_OLS_SGD = sgd.ADAM(
                x = X_train, 
                y = f_train, 
                startWeights = weights_OLS_SGD, 
                numBatches = numBatches, 
                numEpochs = epochsPerIteration,
                learningRate = learningRates[i], 
                firstMomentFactor= 0.9,
                secondMomentFactor= 0.999,
                costFunction = sgd.cost_OLS_derivative,
            )
            mse_adam[i, j] = dataFunctions.mse(f_test, X_test @ weights_OLS_SGD)

    fig, ax  = plt.subplots(1, 2)
    fig.suptitle('Minibatch size = ' + str(batchSize), fontsize = 14)
    fig.tight_layout()
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax[0].xaxis.set_major_locator(MultipleLocator(1000))
    ax[1].xaxis.set_major_locator(MultipleLocator(1000))
    ax[0].xaxis.set_minor_locator(MultipleLocator(500))
    ax[1].xaxis.set_minor_locator(MultipleLocator(500))
    #ax[0].yaxis.set_major_locator(MultipleLocator(0.01))
    #ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for a in ax:
        a.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        a.tick_params(axis='x', top=True, direction = 'in', length = 10)
        a.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        a.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('Number of epochs')
    ax[1].set_xlabel('Number of epochs')
    ax[0].set_title(r'RMSprop', fontsize = 14)
    ax[1].set_title(r'ADAM', fontsize = 14)

    x_axis = np.linspace(1, totEpochs, numIterations)
    colors = ['r', 'b', 'g', 'darkorange', 'hotpink', 'teal', 'navy']
    for i in range(len(learningRates)):
        ax[0].semilogy(x_axis, mse_rms[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
    for i in range(len(learningRates)- 1, -1, -1):
        ax[1].plot(x_axis, mse_adam[i, :], color = colors[i], label = r'$\gamma = 10^{%i}$' %int(np.log10(learningRates[i]))) 
    for a in ax:
        a.axhline(y = mse_sklearn, color = 'k', label = 'sklearn OLS')
    ax[0].legend(fontsize = 12, loc = 7)
    #plt.savefig('tunerates_1.pdf', bbox_inches = 'tight')
    plt.show()




##################################
#--------------------------------#
#    Execute below functions     #
#   to compute report figures   #
#--------------------------------#
##################################


sgd_main(64)
sgd_main(128)
sgd_ridge(64)
adaptiveGradient(128)
tuneableRates(128)