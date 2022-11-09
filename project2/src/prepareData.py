from statistics import fmean
import numpy as np
from sklearn.model_selection import train_test_split

# import dataFunctions
import franke

##########################################
#  These functions create data, scales,  #
#   Splits and makes it ready for reg    #
##########################################

def createData(N, fMean, fStd):
    """
    Creates data 

    Param
    ---------------
    N : int - len of data
    fMean : float - mean of normal noise in franke
    fStd : float - std of normal noise in franke
    Returns 
    ---------------
    x, y : 1darrays - x, y grid
    fdata : 1darray - function data on the form f(x, y)
    """
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xGrid, yGrid = np.meshgrid(x, y)
    x = xGrid.flatten()
    y = yGrid.flatten()
    # noisy franke function
    fData = franke.noisy_franke(xGrid, yGrid, fMean, fStd, N).flatten()
    return x, y, fData

def scaleData(x, y, fData):
    """
    Scales the data by subtracting mean and dividing by std
    Param
    ---------------
    x, y : 1darrays - x, y grid
    fdata : 1darray - function data on the form f(x, y)

    Returns
    ---------------
    Scaled versions of input data

    """
    xMean = np.mean(x)
    yMean = np.mean(y)
    fMean = np.mean(fData)
    xstd = np.std(x)
    ystd = np.std(y)
    fstd = np.std(fData)

    xScaled = (x - xMean) / xstd
    yScaled = (x - yMean) / ystd
    fDataScaled = (fData - fMean) / fstd

    return xScaled, yScaled, fDataScaled

def shuffleData(data):
    """
    Shuffles an array by making one random permutation of the indeces

    Param
    ---------------
    data : array like - data to shuffle

    Returns
    ---------------
    Shuffled version of data

    """
    rng = np.random.default_rng()
    return rng.permutation(data)

def generateDesignMatrix(x, y, degree):
    """
    Copy-pasted from dataFunctions.py / project1 to have everything in one place
    creates design matrix of degree out of x and y arrays
    """
    n = len(x)
    m = int((degree + 1)*(degree + 2)/ 2)
    X = np.ones((n, m))
    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = x**i * y**j
            p += 1
    return X

def splitData(X, fData, test_size):
    """
    Splits designmatrix and fData into train / test sets

    Param
    ---------------
    X : 2darray  - design matrix
    fData : 1darray - data 
    test_size : float - percentege to use as test. Standard = 0.1

    Returns
    ---------------
    Splitted versions of data and X
    X_train, X_test : 2darray
    f_train, f_test : 1darray

    """
    X_train, X_test, f_train, f_test = train_test_split(X, fData, test_size=test_size)

    return X_train, X_test, f_train, f_test


def initializeData(N, fMean, fStd, deg, test_size):
    """
    Uses the above functions to initialize all data
    """
    #Create data
    x, y, fData = createData(N, fMean, fStd)
    #Scale data
    x, y, fData = scaleData(x, y, fData)
    #Permutation of fData   
    fData = shuffleData(fData)
    #Generate design matrix
    X = generateDesignMatrix(x, y, deg)
    #Split data
    X_train, X_test, f_train, f_test = splitData(X, fData, test_size)
    #Return all data
    return x, y, fData, f_train, f_test, X, X_train, X_test

