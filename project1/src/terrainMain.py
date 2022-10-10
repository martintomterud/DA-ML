#libraries
from contextlib import redirect_stderr
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
np.random.seed(seed=int(42)) 


#src
import regressionClass
import designMatrix
import bootstrap
import franke
import dataFunctions


def mainOLS():
    #path for storing figures
    cd = os.path.abspath('..')
    figpath = cd + "/figures/"
    datapath = cd + '/data/'
    dataname = 'SRTM_data_Norway_1.tif'
    
    # Set up start parameters
    
    #reg method
    method = 'ols'

    #terraindata
    size = 1000 # M x M size of terrain region
    terrain = dataFunctions.importTerrain(datapath + dataname)
    terrainRegion = dataFunctions.terrainGridRegion(terrain, size)
    scaledTerrainRegion = dataFunctions.scaleTerrain(terrainRegion)
    """
    Testing:
    
    plt.figure()
    plt.imshow(terrain)
    plt.figure()
    plt.imshow(terrainRegion)
    plt.figure()
    plt.imshow(scaledTerrainRegion)
    plt.show()
    """
    #set up axes grid and terrain grid
    # axes 
    N = int(100) 
    red = int(size/N) 
    terrainData = scaledTerrainRegion[::red, ::red]
    x = np.linspace(0,1, np.shape(terrainData)[0])
    y = np.linspace(0,1, np.shape(terrainData)[1])
    xGrid, yGrid = np.meshgrid(x, y)
    x = xGrid.flatten()
    y = yGrid.flatten()
    terrainData = terrainData.flatten()
    
    """
    plt.figure()
    plt.imshow(scaledTerrainRegion)
    plt.figure()
    plt.imshow(terrainData)
    plt.show()
    """

    #polynomial degree and array
    maxDegree = 12
    degrees = np.arange(1, maxDegree + 1)
    #containers for storing statistics
    #regular run
    mse_5 = np.zeros(maxDegree)
    mse_10 = np.zeros(maxDegree)

    #run
    for i in range(len(degrees)):
        deg = degrees[i]

        # Regression via k-fold Cross validation with k = 5
        cvRegObj = regressionClass.Regression(method)
        CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, terrainData, method, lamb = 0, deg = deg, k = 5)
        mse_5[i] = CVStatistics[0]


        # Regression via k-fold cv with k = 10
        cvRegObj = regressionClass.Regression(method)
        CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, terrainData, method, lamb = 0, deg = deg, k = 10)
        mse_10[i] = CVStatistics[0]
    
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', top=True, direction = 'in', length = 10)
    ax.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    ax.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax.set_xlabel(r'Polynomial degree $p$')
    ax.set_ylabel(r'MSE')
    ax.plot(degrees, mse_5, color = 'blue', lw = 2, alpha = 0.8, marker = '^', mfc = 'none', label = r'$k = 5$')
    ax.plot(degrees, mse_10, color = 'red', lw = 2, alpha = 0.8, marker = 'v', mfc = 'none', label = r'$k = 10$')
    ax.legend(frameon = False)
    ax.set_title('OLS')
    #plt.savefig(figpath+'terrain_unscaled_ols.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()

def mainRL():
    #path for storing figures
    cd = os.path.abspath('..')
    figpath = cd + "/figures/"
    datapath = cd + '/data/'
    dataname = 'SRTM_data_Norway_1.tif'
    
    # Set up start parameters
    
    #reg method
    method = 'ridge'

    #terraindata
    size = 1000 # M x M size of terrain region
    terrain = dataFunctions.importTerrain(datapath + dataname)
    terrainRegion = dataFunctions.terrainGridRegion(terrain, size)
    scaledTerrainRegion = dataFunctions.scaleTerrain(terrainRegion)
    #set up axes grid and terrain grid
    # axes 
    N = int(100)
    red = int(size/N) 
    terrainData = scaledTerrainRegion[::red, ::red]
    x = np.linspace(0,1, np.shape(terrainData)[0])
    y = np.linspace(0,1, np.shape(terrainData)[1])
    xGrid, yGrid = np.meshgrid(x, y)
    x = xGrid.flatten()
    y = yGrid.flatten()
    terrainData = terrainData.flatten()

    #polynomial degree and array
    maxDegree = 18
    degrees = np.arange(1, maxDegree + 1)
    #lambda array 
    lambdas = np.logspace(-1, -7, 7)
    #lambdas = [0.01]
    lambdas[0] = 0
    numlam = len(lambdas)
    #containers for storing statistics
    #regular run
    mse_5 = np.zeros((numlam, maxDegree))
    mse_10 = np.zeros((numlam, maxDegree))

    #run
    for i in range(numlam):
        for j in range(maxDegree):
            deg = degrees[j]
            lamb = lambdas[i]

            # Regression via k-fold Cross validation with k = 5
            CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, terrainData, method, lamb = lamb, deg = deg, k = 5)
            mse_5[i, j] = CVStatistics[0]


            # Regression via k-fold cv with k = 10
            CVStatistics, cvBetas = dataFunctions.crossValidation(x, y, terrainData, method, lamb = lamb, deg = deg, k = 10)
            mse_10[i, j] = CVStatistics[0]
    
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
    #axs[1].yaxis.set_major_locator(MultipleLocator(0.01))
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
        axs[0].plot(degrees, mse_5[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
        axs[1].plot(degrees, mse_10[i, :], color = colors[i], marker = markers[i], mfc = 'none', lw = 2, alpha = 0.8, label = r'$\lambda = $' +str(lambdas[i]))
    
    axs[0].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[1].legend(frameon = False, ncol = 2, fontsize = 12)
    axs[0].annotate(r'$k = 5$', (2, 0.01), fontsize = 12)
    axs[1].annotate(r'$k = 10$', (2, 0.01), fontsize = 12)
    axs[0].set_title('Lasso')
    #plt.savefig(figpath+'terrain_scaled_lasso.pdf', dpi = 800, bbox_inches = 'tight')
    plt.show()  
    
#mainOLS()
mainRL()