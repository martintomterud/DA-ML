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
#np.random.seed(seed=int(time.time())) 

# set const seed for debugging
np.random.seed(seed=int(2020)) 


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

def mainRL():
    #path for storing figures
    cd = os.path.abspath('.')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'ridge'
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
    lambdas = np.logspace(0, -6, 7)
    lambdas[0] = 0

    #containers for storing statistics
    numLam = len(lambdas)
    numDeg = len(degrees)

    mseBoot = np.zeros((numLam, numDeg))
    mseCV = np.zeros((numLam, numDeg))

    # start loop

    for i in range(len(lambdas)):
        for j in range(len(degrees)):
            lamb = lambdas[i]
            deg = degrees[j]

            # Regression via bootstrap
            bootRegObj = regressionClass.Regression(method)
            bootRegObj.setLamb(lamb)
            bootStatistics = dataFunctions.bootstrap_arrays(x, y, frankeData, bootRegObj, deg, lamb = lamb, test_size = 0.2, k = 20)
            mseBoot[i, j] = bootStatistics[0]
        
            # Regression via k-fold Cross validation with k = 5
            cvRegObj = regressionClass.Regression(method)
            cvRegObj.setLamb(lamb)
            CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, frankeData, method, lamb = lamb, deg = deg, k = 5)
            mseCV[i, j] = CVStatistics[0]

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
        axs[0].plot(degrees, mseBoot[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
        axs[1].plot(degrees, mseCV[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
    
    axs[0].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[1].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[0].annotate('Bootstrap', (3, 0.03), fontsize = 12)
    axs[1].annotate(r'$k = 5$ fold CV', (3, 0.03), fontsize = 12)

    plt.savefig(figpath+'mseRidge_n20.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()


mainRL()