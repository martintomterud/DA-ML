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
#src
import regressionClass
import designMatrix
import bootstrap
import franke
import dataFunctions

def frankeOLS(maxdegree):
    method = 'ols'
    N_points  = 1000
    x = np.random.rand(N_points)
    y = np.random.rand(N_points)
    yData = franke.franke(x, y)
    degrees = np.arange(1, maxdegree + 1)

    mse = np.zeros(maxdegree)
    r2 = np.zeros(maxdegree)

    for i in range(len(degrees)):
        deg = degrees[i]
        desMat = designMatrix.designMatrix(N_points, deg)
        desMat.generateDesignMatrix(x, y, deg)
        X = desMat.X
        regObject = regressionClass.Regression(method)
        regObject.computeRegression(X, yData)
        regObject.makePrediction()
        regObject.stat_mse()
        regObject.stat_r2()
        mse[i] = regObject.mse
        r2[i] = regObject.r2 

    fig1, ax1  = plt.subplots()
    plt.xlabel(r'Polynomial degree $p$')
    plt.ylabel('MSE')
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    ax1.tick_params(axis='x', top=True, direction = 'in', length = 10)
    ax1.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    ax1.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax1.plot(degrees, mse, color = 'red', linewidth = 2, alpha = 0.8, marker = 'o')
    ax1.set_yscale('log')

    fig2, ax2  = plt.subplots()
    plt.xlabel(r'Polynomial degree $p$')
    plt.ylabel('R2')
    #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    ax2.tick_params(axis='x', top=True, direction = 'in', length = 10)
    ax2.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    ax2.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax2.plot(degrees, r2, color = 'blue', linewidth = 2, alpha = 0.8, marker = 'o')
    ax2.set_yscale('log')
    plt.show()




frankeOLS(5)