from re import T
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.utils import shuffle, resample
import random 
import designMatrix
import regressionClass
from imageio import imread


def meshgrid(N):
    """
    Set up meshgrid of n points
    in x and y defined on [0, 1]x[0, 1]

    Paramter
    ---------
    N : int
        number of points on [0, 1]
    
    Returns
    ---------
    x,y : ndarray
        array of type (x, y) indexed with double indeces
    
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    return np.meshgrid(x, y)

#Functions for computing data statistics

def mse(data, prediction):
    return np.mean((data - prediction)**2)

def r2(data, prediction):
    mean = np.mean(data)
    r2 = 1 - np.sum((data- prediction)**2) / np.sum((data - mean)**2)        
    return r2 

def variance(data):
    return mse(data, np.mean(data))

def bias(data, prediction):
    return mse(data, np.mean(prediction))

def generateDesignMatrix(x, y, degree):
    n = len(x)
    m = int((degree + 1)*(degree + 2)/ 2)
    X = np.ones((n, m))
    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = x**i * y**j
            p += 1
    return X

def splitTrainTest(X, y, method, test_size):
    """
    uses sklearn functionality to split data into testing and training
    performs linear regression on data and computes a prediction

    Parameters
    -----------------
    method : string
                The regression case to be executed
                Available methods:
                    'ols', 'ridge', 'lasso'
    
    test_size : float between 0 and 1
                the size of the test data (usually about 0.2)
    """

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    #regression on train data
    regObject = regressionClass.Regression(method)
    regObject.computeRegression(X_train, y_train)
    #prediction on test data
    yPrediction = regObject.returnPrediction(X_test)
    #statistics
    mse = np.mean((y_test - yPrediction)**2)
    R2 = r2(yPrediction, y_test)
    bias = np.mean((y_test - np.mean(yPrediction))**2)
    variance = np.var(yPrediction)

    return [mse, R2, bias, variance]

def bootstrap(X, y, regressionObject, test_size, k):
    """
    Performs the bootstrap  resampling algorithm k times
    Uses the regression object from initialization
    Assumes regression already done

    Parameters
    ---------------
    k : int
        number of "folds", i.e. how many times we perform the algorithm
    test_size : float
        size of test data for use in scikit learn train_test_split
    
    Algorithm follows the bootstrap algorithm on this link:
    https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
    but loops over all k fold values to compute beta statistics

    Computes statistics on predictions and betas and sets them as class variables.
    
    """
    # split data from initialised model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    N = X_train.shape[0]
    N_betas = np.shape(X)[1]
    
    #storing computed quantities
    yPredictions = np.empty((y_test.shape[0], k))
    BETAS = np.zeros((N_betas, k))

    #run k bootstrapping loops
    for i in range(k):
        #Fit model to random indices
        idx = np.random.randint(0, N, N)
        X_idx, y_idx = X_train[idx], y_train[idx]
        regressionObject.computeRegression(X_idx, y_idx)

        # Make prediction and save data 
        yPredictions[:, i] = regressionObject.returnPrediction(X_test).ravel()
        BETAS[:, i] = regressionObject.beta
    
    # reshape y_test into matrix do compare with each prediction:
    yTest = np.reshape(y_test, (len(y_test), 1))

    #compute statistics

    MSE = np.mean( np.mean((yTest - yPredictions)**2, axis=1, keepdims=True) )
    R2 = np.mean(1 - np.sum((yTest - yPredictions)**2, axis=1, keepdims=True)/np.sum((yTest - np.mean(yTest))**2, axis=1, keepdims=True) )
    BIAS = np.mean( (yTest - np.mean(yPredictions, axis=1, keepdims=True))**2 )
    VAR = np.mean( np.var(yPredictions, axis=1, keepdims=True) )
    return [MSE, R2, BIAS, VAR]
    
def bootstrap_arrays(x, y, yData, regressionObject, deg, lamb, test_size, k):
    """
    Variation of the function above taht takes all arrays and generates 
    design matrix from the function, not the class
    """

    #split data
    x_train, x_test, y_train, y_test, yData_train, yData_test = train_test_split(x, y, yData, test_size=test_size)
    #test design matrix
    X_test = generateDesignMatrix(x_test, y_test, deg)
    #storing computed quantities
    yDataPredictions = np.empty((yData_test.shape[0], k))
    N = len(x_train)

    #bootstaping loops
    for i in range(k):
        idx = np.random.randint(0, N, N)
        x_idx, y_idx, yData_idx = x_train[idx], y_train[idx], yData_train[idx]
        #x_idx, y_idx, yData_idx = resample(x_train, y_train, yData_train)
        X = generateDesignMatrix(x_idx, y_idx, deg)
        regressionObject.computeRegression(X, yData_idx)
        yData_prediction = regressionObject.returnPrediction(X_test)
        yDataPredictions[:, i] = yData_prediction
     # reshape y_test into matrix do compare with each prediction:
    yDataTest = np.reshape(yData_test, (len(yData_test), 1))

    #compute statistics

    MSE = np.mean( np.mean((yDataTest - yDataPredictions)**2, axis=1, keepdims=True) )
    R2 = np.mean(1 - np.sum((yDataTest - yDataPredictions)**2, axis=1, keepdims=True)/np.sum((yDataTest - np.mean(yDataTest))**2, axis=1, keepdims=True) )
    BIAS = np.mean( (yDataTest - np.mean(yDataPredictions, axis=1, keepdims=True))**2 )
    VAR = np.mean( np.var(yDataPredictions, axis=1, keepdims=True) )
    return [MSE, R2, BIAS, VAR]
    

def crossValidation(x, y, yData, method, lamb, deg, k):
    
    """
    performs k-fold cross validation on coordinate specified by x and y and data specified by y data

        Parameters
    -----------------
    x: 1d array
        array of coordinates along x
    
    y: 1d array
        array of coordinates along y

    yData: 1d array
        array of data points
    
    method : string
            regression method to use (ols, ridge or lasso)
    
    lamb : float
            lambda parameter for ridge / lasso. Ols => lamb = 0.0

    deg : int
            degree of polynomials

    k : int 
        number of folds
    
    Returns
    --------------------

    statistics on computed resampling:
    [mse, r2, var, bias], betas
    the statistics as an array, and a mtrix of computed betas
    """

    #initialize containers for storing computed values
    m = int((deg + 2)*(deg + 1) / 2)
    MSE = np.zeros(k)
    R2 = np.zeros(k)
    VAR = np.zeros(k)
    BIAS = np.zeros(k)
    BETAS = np.zeros((m, k))
    # Split arrays into k folds of equal size q
    q = len(x) // k
    
    #shuffle in data arrays randomly
    #create array copies for shuffling
    #use np.random.permutation to shuffle arrays in unison
    shuffler = np.random.permutation(len(x)) 
    xShuf = x[shuffler]
    yShuf = y[shuffler]
    yDataShuf = yData[shuffler]

    #split shuffled data into k folds 
    xSplit = np.array_split(xShuf, k)
    ySplit = np.array_split(yShuf, k)
    yDataSplit = np.array_split(yDataShuf, k)
   
    #start loop over folds
    for i in range(k):
        #extract test data from the splitted arrays
        #then delete the test data from the arrays to get
        #the train data
        xTest = xSplit[i]
        xTrain = np.delete(xSplit, i, axis = 0).ravel()
        yTest = ySplit[i]
        yTrain = np.delete(ySplit, i, axis = 0).ravel()
        yDataTest = yDataSplit[i]
        yDataTrain = np.delete(yDataSplit, i, axis = 0).ravel()

        ### FITTING MODEL TO TRAIN DATA ###
        X_Train = generateDesignMatrix(xTrain, yTrain, deg)
        #copmute regression from specified model
        
        regObject = regressionClass.Regression(method)
        regObject.setLamb(lamb)
        regObject.computeRegression(X_Train, yDataTrain)
        beta = regObject.beta

        ### SET UP DESMAT FOR TEST DATA ###
        X_Test = generateDesignMatrix(xTest, yTest, deg)

        ### MAKE FIT OF THE MODELLED BETA ON TEST MATRIX ###
        yDataFit = regObject.returnPrediction(X_Test)

        #Compute statistics and append to containers
        MSE[i] = mse(yDataTest, yDataFit)
        R2[i] = r2(yDataTest, yDataFit)
        BIAS[i] = bias(yDataTest, yDataFit)
        VAR[i] = variance(yDataFit)
        BETAS[:, i] = beta

    #return mean of statistics and beta matrix

    return [np.mean(MSE), np.mean(R2), np.mean(BIAS), np.mean(VAR)], BETAS
    


def importTerrain(file):
    """
    Imports the terrain file and returns data
    """
    terrain = imread(file)
    return terrain

def terrainGridRegion(terrain, size):
    """
    Picks out random size x size region in the terrain grid
    """
    xlen = np.shape(terrain)[0]
    ylen = np.shape(terrain)[1]
    x_rand = np.random.randint(0, xlen - size)
    y_rand = np.random.randint(0, ylen - size)
    region = terrain[x_rand : x_rand + size, y_rand : y_rand + size]
    return region
    
def scaleTerrain(terrain):
    """
    Scales terrain data by first subtracting minimum
    and the normalizes by dividing by max
    """
    max = np.amax(terrain)
    min = np.amin(terrain)
    t = np.copy(terrain).astype(np.float64)
    T = (t - min) / (max - min)
    return T

