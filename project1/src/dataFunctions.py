import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.utils import shuffle
import designMatrix
import regressionClass

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

def splitTrainTest(self, method, test_size):
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
    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
    #regression on train data
    regObject = regressionClass.Regression(method)
    regObject.computeRegression(X_train, y_train)
    #prediction on test data
    yPrediction = regObject.returnPrediction(X_test)
    #statistics
    mse = np.mean((y_test - yPrediction)**2)
    r2 = r2_score(y_test, yPrediction)
    bias = np.mean((y_test - np.mean(yPrediction))**2)
    variance = np.var(yPrediction)

    return mse, r2, bias, variance

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
    mse = np.zeros(k)
    r2 = np.zeros(k)
    var = np.zeros(k)
    bias = np.zeros(k)
    BETAS = np.zeros((m, k))

    #shuffle indeces using sklearn shuffle funtion
    xShuf, yShuf, yDataShuf = shuffle(x, y, yData)

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
        # set up design matrix for training data
        desMatTrain = designMatrix(len(xTrain), deg)
        # populate the design matrix
        desMatTrain.generateDesignMatrix(xTrain, yTrain, deg)
        #return th design matrix
        X_Train = desMatTrain.returnMatrix()
        #copmute regression from specified model
        regObject = regressionClass.Regression(method)
        regObject.setLamb(lamb)
        regObject.computeRegression(X_Train, yDataTrain)
        beta = regObject.beta

        ### SET UP DESMAT FOR TEST DATA ###
        desMatTest = designMatrix(len(xTest), deg)
        # populate the design matrix
        desMatTest.generateDesignMatrix(xTest, yTest, deg)
        #return th design matrix
        X_Test = desMatTest.returnMatrix()

        ### MAKE FIT OF THE MODELLED BETA ON TEST MATRIX ###
        yDataFit = regObject.returnPrediction(X_Test)

        #Compute statistics and append to containers
        mse[i] = mse(yDataTest, yDataFit)
        r2[i] = r2(yDataTest, yDataFit)
        bias[i] = bias(yDataTest, yDataFit)
        var[i] = var(yDataFit)
        BETAS[:, i] = beta

    #return mean of statistics and beta matrix

    return [np.mean(mse), np.mean(r2), np.mean(bias), np.mean(var)], BETAS
    




