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

#set seed for numpy random using time 
import time 
np.random.seed(seed=int(time.time())) 

#src
import regressionClass
import designMatrix
import bootstrap
import franke
import dataFunctions

"""
This file executes the code and functions used to generate the results for the ols method
"""

def main_OLS():
    # Set up start parameters
    #reg method
    method = 'ols'
    # axes 
    N = int(30) 
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
    #containers for storing statistics
    #regular run
    mse = np.zeros(maxDegree)
    r2 = np.zeros(maxDegree)
    #bootstrap run
    mseBoot = np.zeros(maxDegree) 
    r2Boot = np.zeros(maxDegree)
    biasBoot = np.zeros(maxDegree)
    varBoot = np.zeros(maxDegree)

    #crossvalidation
    mseCV = np.zeros(maxDegree) 
    r2CV = np.zeros(maxDegree)
    biasCV = np.zeros(maxDegree)
    varCV = np.zeros(maxDegree)

    #crossvalidation
    mseCV_2 = np.zeros(maxDegree) 
    r2CV_2 = np.zeros(maxDegree)
    biasCV_2 = np.zeros(maxDegree)
    varCV_2 = np.zeros(maxDegree)

    # start run
    for i in range(len(degrees)):
        deg = degrees[i]
        # Design matrix
        desMat = designMatrix.designMatrix(N, deg)
        desMat.generateDesignMatrix(x, y, deg)
        X = desMat.X

        # Regression without splitting data
        regObject = regressionClass.Regression(method)
        regObject.computeRegression(X, frankeData)
        regObject.makePrediction()
        regObject.stat_mse()
        regObject.stat_r2()
        mse[i] = regObject.mse
        r2[i] = regObject.r2

        # Regression via bootstrap
        bootRegObj = regressionClass.Regression(method)
        bootStatistics = dataFunctions.bootstrap_arrays(x, y, frankeData, bootRegObj, deg, lamb = 0, test_size = 0.2, k = 100)
        mseBoot[i] = bootStatistics[0]
        r2Boot[i] = bootStatistics[1]
        biasBoot[i] = bootStatistics[2]
        varBoot[i] = bootStatistics[3]

        # Regression via k-fold Cross validation with k = 5
        cvRegObj = regressionClass.Regression(method)
        CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, frankeData, method, lamb = 0, deg = deg, k = 5)
        mseCV[i] = CVStatistics[0]
        r2CV[i] = CVStatistics[1]
        biasCV[i] = CVStatistics[2]
        varCV[i] = CVStatistics[3]

        # Regression via k-fold cv with k = 10
        cvRegObj = regressionClass.Regression(method)
        CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, frankeData, method, lamb = 0, deg = deg, k = 10)
        mseCV_2[i] = CVStatistics[0]
        r2CV_2[i] = CVStatistics[1]
        biasCV_2[i] = CVStatistics[2]
        varCV_2[i] = CVStatistics[3]

    #set up figure parameters
    #Canvas
    fig, axs = plt.subplots(2, 1)
    axs[1].sharex(axs[0])
    plt.subplots_adjust(hspace=.0)
    #Ticks
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))
    for ax in axs:
        ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
        ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    #Labels
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('R2')
    axs[1].set_xlabel(r'Polynomial degree $p$')
    #Plotting
    #ordinary
    axs[0].plot(degrees, mse, color = 'r', lw = 2, marker = 'o', alpha = 0.8)
    axs[1].plot(degrees, r2, color = 'r', lw = 2, marker = 'o', alpha = 0.8)
    #boot
    axs[0].plot(degrees, mseBoot, color = 'blue', lw = 2, alpha = 0.8, marker = '^')
    #axs[1].plot(degrees, r2Boot, color = 'blue', lw = 2, alpha = 0.8, marker = '^')
    #cv
    axs[0].plot(degrees, mseCV, color = 'green', lw = 2, alpha = 0.8, marker = 'v')
    axs[1].plot(degrees, r2CV, color = 'green', lw = 2, alpha = 0.8, marker = 'v')
    axs[0].plot(degrees, mseCV_2, color = 'magenta', lw = 2, alpha = 0.8, marker = 'v', mfc = 'none')
    axs[1].plot(degrees, r2CV_2, color = 'magenta', lw = 2, alpha = 0.8, marker = 'v', mfc = 'none')

    plt.show() 

    fig, ax = plt.subplots()

    plt.plot(degrees, mseBoot)
    plt.plot(degrees, varBoot, ls = 'dashed')
    plt.plot(degrees, biasBoot, ls = 'dotted')
    plt.plot(degrees, mseCV)
    plt.plot(degrees, varCV, ls = 'dashed')
    plt.plot(degrees, biasCV, ls = 'dotted')
    plt.show()

main_OLS()