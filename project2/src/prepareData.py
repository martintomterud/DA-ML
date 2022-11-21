from statistics import fmean
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    fData = franke.noisy_franke(x, y, fMean, fStd, N)
    return x, y, fData

def createDataNoiseScale(N, scale):
    """
    Creates data 

    Param
    ---------------
    N : int - len of data
    scale : float - scale of noise added to franke
    Returns 
    ---------------
    x, y : 1darrays - x, y grid
    fdata : 1darray - function data on the form f(x, y)
    """
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    fData = franke.noise_scale_franke(x, y, scale, N)
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
    n = len(data)
    perm = np.random.permutation(n)
    return(data[perm])

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

def splitDataDM(X, fData, test_size):
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

def splitData(x, y, fData, test_size):
    """
    Splits x and y coords and fData into train / test sets

    Param
    ---------------
    x, y : 1darray  - coord data axes
    fData : 1darray - data 
    test_size : float - percentege to use as test. Standard = 0.1

    Returns
    ---------------
    Splitted versions of data and X
    X_train, X_test : 2darray
    f_train, f_test : 1darray

    """
    x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(x, y, fData, test_size=test_size)

    return x_train, x_test, y_train, y_test, f_train, f_test

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
    X_train, X_test, f_train, f_test = splitDataDM(X, fData, test_size)
    #Return all data
    return x, y, fData, f_train, f_test, X, X_train, X_test

def center_fData(fTrain, fTest):
    """
    Centers splitted function data by subtracting mean of train data
    
    Param
    ---------------
    fTrain, fTest: splitted data to fit

    Returns
    ---------------
    Centered data
    fTrain, fTest : 1darray
    """
    fMean = np.mean(fTrain)
    fTrain -= fMean
    fTest -= fMean
    return fTrain, fTest

def scaleDesignMatrix(X_train, X_test):
    """
    Uses scikit learns scale to scale teh design matrix and prep for 
    sgd minimization
    
    """
    scalerFunction = StandardScaler()
    scalerFunction.fit(X_train)
    X_train_scaled = scalerFunction.transform(X_train)
    X_test_scaled = scalerFunction.transform(X_test)
    X_train_scaled = X_train_scaled[:, 1:]
    X_test_scaled = X_test_scaled[:, 1:]
    return X_train_scaled, X_test_scaled

def prepForSGD(N, fMean, fStd, degree):
    """
    Prepares franke function and design matrix from
    a set of intial conditions specified 
    """
    x, y, f = createData(N, fMean, fStd)
    x_train, x_test, y_train, y_test, f_train, f_test = splitData(x, y, f, test_size=0.2)
    f_train, f_test = center_fData(f_train, f_test)
    X_train = generateDesignMatrix(x_train, y_train, degree)
    X_test = generateDesignMatrix(x_test, y_test, degree)
    X_train, X_test = scaleDesignMatrix(X_train, X_test)
    return X_train, X_test, f_train, f_test

def prepForSGD_Scale(N, scale, degree):
    """
    Prepares franke function and design matrix from
    a set of intial conditions specified 
    """
    x, y, f = createDataNoiseScale(N, scale)
    x_train, x_test, y_train, y_test, f_train, f_test = splitData(x, y, f, test_size=0.2)
    f_train, f_test = center_fData(f_train, f_test)
    X_train = generateDesignMatrix(x_train, y_train, degree)
    X_test = generateDesignMatrix(x_test, y_test, degree)
    X_train, X_test = scaleDesignMatrix(X_train, X_test)
    return X_train, X_test, f_train, f_test



def generate_data(n, function, noise=False, mean_noise=0, std_noise=1):
    
    rng = np.random.default_rng()

    match function:
        case "franke":
            x = rng.random(size=(n,2))
            y = franke.franke(x[:,0], x[:,1]).reshape(-1, 1)
        case "polynomial":
            x = rng.uniform(-0.5, 1.2, size=(n,1))
            # coefs = rng.standard_normal(3)
            # y = coefs[0] * .1 * x + coefs[1] * x**2 + coefs[2] * x**3 + .02*rng.standard_normal((n,1))
            # y = .1*x + x**2 - x**3 + .02*rng.standard_normal((n,1))
            y = .1*x + x**2 - x**3
        case _:
            raise NameError("The available functions are \"franke\" and \"polynomial\".")

    if noise:
        y += rng.normal(mean_noise, std_noise, size=y.shape)

    return x, y

def prepare_regression(x, y):
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=.2)

    scale_x = StandardScaler().fit(x_tr)
    x_tr = scale_x.transform(x_tr)
    x_te = scale_x.transform(x_te)

    scale_y = StandardScaler().fit(y_tr)
    y_tr = scale_y.transform(y_tr)
    y_te = scale_y.transform(y_te)

    return x_tr, x_te, y_tr, y_te

def prepare_classification(x, y):
    y = y.reshape(-1, 1)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=.2)

    scale_x = StandardScaler().fit(x_tr)
    x_tr = scale_x.transform(x_tr)
    x_te = scale_x.transform(x_te)

    return x_tr, x_te, y_tr, y_te
