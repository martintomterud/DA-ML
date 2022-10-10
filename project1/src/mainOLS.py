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
#np.random.seed(seed=int(42)) 


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
    #path for storing figures
    cd = os.path.abspath('.')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'ols'
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
    #containers for storing statistics
    #regular run
    mse = np.zeros(maxDegree)
    r2 = np.zeros(maxDegree)
    #split run
    mseSplit = np.zeros(maxDegree)
    r2Split = np.zeros(maxDegree)
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

        # Regression on splitted data
        X = dataFunctions.generateDesignMatrix(x, y, deg)
        splitStat = dataFunctions.splitTrainTest(X, frankeData, method, test_size= 0.2)
        mseSplit[i] = splitStat[0]
        r2Split[i] = splitStat[1]


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
    axs[0].xaxis.set_major_locator(MultipleLocator(2))
    axs[1].xaxis.set_major_locator(MultipleLocator(2))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.01))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for ax in axs:
        ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
        ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    #Labels
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel(r'$R^2$')
    axs[1].set_xlabel(r'Polynomial degree $p$')
    #Plotting
    #ordinary
    axs[0].plot(degrees, mseSplit, color = 'r', lw = 2, marker = 'o', alpha = 0.6, label = 'only split')
    axs[1].plot(degrees, r2Split, color = 'r', lw = 2, marker = 'o', alpha = 0.6)
    #boot
    axs[0].plot(degrees, mseBoot, color = 'blue', lw = 2, alpha = 0.6, marker = '^', label = 'bootstrap')
    #axs[1].plot(degrees, r2Boot, color = 'blue', lw = 2, alpha = 0.8, marker = '^')
    #cv
    axs[0].plot(degrees, mseCV, color = 'green', lw = 2, alpha = 0.6, marker = 'v', label = r'$k = 5$-fold CV')
    axs[1].plot(degrees, r2CV, color = 'green', lw = 2, alpha = 0.6, marker = 'v')
    axs[0].plot(degrees, mseCV_2, color = 'magenta', lw = 2, alpha = 0.6, marker = 'v', mfc = 'none', label = r'$k = 10$-fold CV')
    axs[1].plot(degrees, r2CV_2, color = 'magenta', lw = 2, alpha = 0.6, marker = 'v', mfc = 'none')
    axs[0].set_ylim(0.005, 0.045)
    axs[0].legend(frameon = False)
    #plt.savefig(figpath+'ols_stat.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show() 
    '''
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
    ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    ax.tick_params(axis='y', right=True, direction = 'in', length = 10)

    plt.plot(degrees, mseBoot, color = 'blue', lw = 2, alpha = 0.8, marker = '^', label = 'MSE')
    plt.plot(degrees, varBoot, ls = 'dashed', color = 'blue', lw = 2, alpha = 0.6, marker = 's', label = 'Variance')
    plt.plot(degrees, biasBoot, ls = 'dotted', color = 'blue', lw = 2, alpha = 0.4, marker = 'o', label = 'Bias')
    plt.legend(frameon = False)
    ax.set_xlabel(r'Polynomial degree $p$')
    ax.set_ylabel(r'value of statistical parameter')
    plt.savefig(figpath+'ols_bias_var.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()
    '''

def var_bias():
    cd = os.path.abspath('.')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'ols'
    # axes 
    # making two different set of axes to show the difference in beta var
    # depending on number of grid points
    N = int(20) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x_1 = xGrid.flatten()
    y_1 = yGrid.flatten()
    # noisy franke function
    fMean = 0
    fStd = 0.1
    frankeData_1 = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()
    N = int(50) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x_2 = xGrid.flatten()
    y_2 = yGrid.flatten()
    # noisy franke function
    frankeData_2 = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()

    # set up for regression loop
    #polynomial degree and array
    maxDegree = 11
    degrees = np.arange(1, maxDegree + 1)
    #containers for storing statistics
    mse_1 = np.zeros(maxDegree)
    bias_1 = np.zeros(maxDegree)
    var_1 = np.zeros(maxDegree)
    mse_2 = np.zeros(maxDegree)
    bias_2 = np.zeros(maxDegree)
    var_2 = np.zeros(maxDegree)
    # start run
    for i in range(len(degrees)):
        deg = degrees[i]
       
        # Regression via bootstrap 1
        bootRegObj = regressionClass.Regression(method)
        bootStatistics = dataFunctions.bootstrap_arrays(x_1, y_1, frankeData_1, bootRegObj, deg, lamb = 0, test_size = 0.2, k = 100)
        mse_1[i] = bootStatistics[0]
        bias_1[i] = bootStatistics[2]
        var_1[i] = bootStatistics[3]
        # Regression via bootstrap 2
        bootRegObj = regressionClass.Regression(method)
        bootStatistics = dataFunctions.bootstrap_arrays(x_2, y_2, frankeData_2, bootRegObj, deg, lamb = 0, test_size = 0.2, k = 100)
        mse_2[i] = bootStatistics[0]
        bias_2[i] = bootStatistics[2]
        var_2[i] = bootStatistics[3]

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
    #axs[0].yaxis.set_major_locator(MultipleLocator(0.01))
    #axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    for ax in axs:
        ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
        ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
        ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
        ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    #Labels
    #axs[0].set_ylabel('Value of statistical parameter')
    #axs[1].set_ylabel('Value of statistical parameter')
    fig.text(0.01, 0.5, 'value of statistical parameter', va='center', rotation='vertical')
    axs[1].set_xlabel(r'Polynomial degree $p$')

    axs[0].plot(degrees, mse_1, color = 'blue', lw = 2, alpha = 0.8, marker = '^', label = 'MSE')
    axs[0].plot(degrees, var_1, ls = 'dashed', color = 'blue', lw = 2, alpha = 0.6, marker = 's', label = 'Variance')
    axs[0].plot(degrees, bias_1, ls = 'dotted', color = 'blue', lw = 2, alpha = 0.4, marker = 'o', label = r'Bias$^2$')
    axs[1].plot(degrees, mse_2, color = 'red', lw = 2, alpha = 0.8, marker = '^', label = 'MSE')
    axs[1].plot(degrees, var_2, ls = 'dashed', color = 'red', lw = 2, alpha = 0.6, marker = 's', label = 'Variance')
    axs[1].plot(degrees, bias_2, ls = 'dotted', color = 'red', lw = 2, alpha = 0.4, marker = 'o', label = r'Bias$^2$')
    
    axs[0].legend(frameon = False)
    axs[1].legend(frameon = False)

    axs[0].set_ylim(-0.005, 0.051)
    axs[1].set_ylim(-0.005, 0.051)

    axs[0].annotate(r'$N = 20 \times 20$', (0.9, 0.041))
    axs[1].annotate(r'$N = 50 \times 50$', (0.9, 0.041))
    
    #plt.savefig(figpath+'ols_bias_var.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()

def beta_var():
    cd = os.path.abspath('..')
    figpath = cd + "/figures/"
    # Set up start parameters
    #reg method
    method = 'ols'
    # deg of pl
    polydeg = 5
    # axes 
    # making two different set of axes to show the difference in beta var
    # depending on number of grid points
    N = int(20) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x_1 = xGrid.flatten()
    y_1 = yGrid.flatten()
    # noisy franke function
    fMean = 0
    fStd = 0.1
    #95 % conf int
    confInt = 1.96
    frankeData_1 = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()
    N = int(50) 
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x_2 = xGrid.flatten()
    y_2 = yGrid.flatten()
    # noisy franke function
    frankeData_2 = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()

    # set 1
    X_1 = dataFunctions.generateDesignMatrix(x_1, y_1, polydeg)
    regOb_1 = regressionClass.Regression(method)
    regOb_1.computeRegression(X_1, frankeData_1)
    beta_1 = regOb_1.beta
    error_1 = fStd*np.sqrt(np.diag( np.linalg.inv( X_1.T @ X_1 )) )

    # set 2
    X_2 = dataFunctions.generateDesignMatrix(x_2, y_2, polydeg)
    regOb_2 = regressionClass.Regression(method)
    regOb_2.computeRegression(X_2, frankeData_2)
    beta_2 = regOb_2.beta
    error_2 = fStd*np.sqrt(np.diag( np.linalg.inv( X_2.T @ X_2 )) )

    # figure

    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
    ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    
    ax.set_xlabel(r'j')
    ax.set_ylabel(r'$\beta_j$')

    ax.errorbar(np.arange(1,len(beta_1)+1), beta_1, yerr=error_1 ,fmt='o', mfc = 'none', capsize = 5, linewidth = 3, alpha = 0.6, color = 'forestgreen', label = r'$N = 20 \times 20$')
    ax.errorbar(np.arange(1,len(beta_2)+1), beta_2, yerr=error_2 ,fmt='s', mfc = 'none', capsize = 5, linewidth = 2, alpha = 0.6, color = 'darkviolet', label = r'$N = 50 \times 50$')
    ax.legend(frameon = False)
    #plt.savefig(figpath+'beta_var.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()

#main_OLS()
#beta_var()
#var_bias()
