#libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import os

#set seed for numpy random using time 
import time 
np.random.seed(seed=int(time.time())) 

# set const seed for debugging
# np.random.seed(seed=int(42)) 


#src
import regressionClass
import designMatrix
import bootstrap
import franke
import dataFunctions


"""
This function works for both ridge and lasso
just change method parameter
and
remember changing name of savefigs
"""

def MSE_RL():
    #path for storing figures
    cd = os.path.abspath('.')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'lasso'
    # axes 
    N = int(50) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x = xGrid.flatten()
    y = yGrid.flatten()
    # noisy franke function
    fMean = 0
    fStd = 0.1
    frankeData = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()

    #polynomial degree and array
    maxDegree = 14
    degrees = np.arange(1, maxDegree + 1)

    #lambda array 
    lambdas = np.logspace(-2, -8, 7)
    #lambdas[0] = 0

    #containers for storing statistics
    numLam = len(lambdas)
    numDeg = len(degrees)

    mseCV_5 = np.zeros((numLam, numDeg))
    mseCV_10 = np.zeros((numLam, numDeg))

    # start loop

    for i in range(len(lambdas)):
        for j in range(len(degrees)):
            lamb = lambdas[i]
            deg = degrees[j]
        
            # Regression via k-fold Cross validation with k = 5
            cvRegObj = regressionClass.Regression(method)
            cvRegObj.setLamb(lamb)
            CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, frankeData, method, lamb = lamb, deg = deg, k = 5)
            mseCV_5[i, j] = CVStatistics[0]

            # Regression via k-fold Cross validation with k = 10
            cvRegObj = regressionClass.Regression(method)
            cvRegObj.setLamb(lamb)
            CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, frankeData, method, lamb = lamb, deg = deg, k = 10)
            mseCV_10[i, j] = CVStatistics[0]

    # set up figure
    # canvas    

    fig, axs = plt.subplots(2, 1)
    axs[1].sharex(axs[0])
    plt.subplots_adjust(hspace=.0)
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    axs[0].xaxis.set_major_locator(MultipleLocator(2))
    axs[1].xaxis.set_major_locator(MultipleLocator(2))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.01))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.01))
    for ax in axs:
        ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
        ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    #Labels
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('MSE')
    axs[1].set_xlabel(r'Polynomial degree $p$')

    colors = ['k', 'red', 'blue', 'green', 'magenta', 'darkorange', 'cyan']
    markers = ['.', 's', 'v', '^', 'o', '*', 'X']

    for i in range(len(lambdas)):
        axs[0].plot(degrees, mseCV_5[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
        axs[1].plot(degrees, mseCV_10[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
    
    axs[0].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[1].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[0].annotate(r'$k = 5$', (2, 0.035), fontsize = 12)
    axs[1].annotate(r'$k = 10$', (2, 0.035), fontsize = 12)
    axs[0].set_title(r'$N = 50 \times 50$')

    #plt.savefig(figpath+'mseLasso_n50.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()

def BI_VAR_RL():
    """
    Remember to change the range onf lambdas bedore running lasso
    It does not handle lamb = 0 well
    """

    #path for storing figures
    cd = os.path.abspath('.')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'lasso'
    # axes 
    N = int(20) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x = xGrid.flatten()
    y = yGrid.flatten()
    # noisy franke function
    fMean = 0
    fStd = 0.1
    frankeData = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()

    #polynomial degree and array
    maxDegree = 14
    degrees = np.arange(1, maxDegree + 1)

    #lambda array 
    lambdas = [0.1, 1e-6]

    #containers for storing statistics
    numLam = len(lambdas)
    numDeg = len(degrees)

    mse = np.zeros((numLam, numDeg))
    bias = np.zeros((numLam, numDeg))
    var = np.zeros((numLam, numDeg))
    
    #looping
    for i in range(len(lambdas)):
        for j in range(len(degrees)):
            lamb = lambdas[i]
            deg = degrees[j]
        
            # Regression via kbootstrap
            regObj = regressionClass.Regression(method)
            regObj.setLamb(lamb)
            bootStatistics = dataFunctions.bootstrap_arrays(x, y, frankeData, regObj, deg = deg, lamb = lamb, test_size= 0.2, k = 10)
            mse[i, j] = bootStatistics[0]
            bias[i, j] = bootStatistics[2]
            var[i, j] = bootStatistics[3]

    #set up figure
    # do one panel with smallest lambda and one with largest lambda
    fig, axs = plt.subplots(2, 1)
    axs[1].sharex(axs[0])
    plt.subplots_adjust(hspace=.0)
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    axs[0].xaxis.set_major_locator(MultipleLocator(2))
    axs[1].xaxis.set_major_locator(MultipleLocator(2))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.03))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.01))
    for ax in axs:
        ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
        ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    #Labels
    fig.text(0.005, 0.5, 'value of statistical parameter', va='center', rotation='vertical')
    axs[1].set_xlabel(r'Polynomial degree $p$')
    
    axs[0].plot(degrees, mse[0, :], color = 'blue', lw = 2, alpha = 0.8, marker = '^', label = 'MSE')
    axs[0].plot(degrees, var[0, :], ls = 'dashed', color = 'blue', lw = 2, alpha = 0.6, marker = 's', label = 'Variance')
    axs[0].plot(degrees, bias[0, :], ls = 'dotted', color = 'blue', lw = 2, alpha = 0.4, marker = 'o', label = r'Bias$^2$')
    axs[1].plot(degrees, mse[1, :], color = 'red', lw = 2, alpha = 0.8, marker = '^', label = 'MSE')
    axs[1].plot(degrees, var[1, :], ls = 'dashed', color = 'red', lw = 2, alpha = 0.6, marker = 's', label = 'Variance')
    axs[1].plot(degrees, bias[1, :], ls = 'dotted', color = 'red', lw = 2, alpha = 0.4, marker = 'o', label = r'Bias$^2$')

    #axs[0].set_ylim(-0.001, 0.06)
    #axs[1].set_ylim(-0.009, 0.06)
    axs[0].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[1].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[0].set_title(r'$N = 20 \times 20$')
    axs[0].annotate(r'$\lambda = $' + str(lambdas[0]), (4, 0.025), fontsize = 12)
    axs[1].annotate(r'$\lambda = $' + str(lambdas[1]), (4, 0.03), fontsize = 12)

    #plt.savefig(figpath+'biasvar_lasso_n20.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()


#MSE_RL()
#BI_VAR_RL()