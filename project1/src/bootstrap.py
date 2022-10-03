import numpy as np
from sklearn.model_selection import train_test_split


import regressionClass
import designMatrix
import dataFunctions

class Bootstrap:

    """
    Bootstrap resampling of regression class
    """

    def __init__(self, regressionObject, X, y):
        self.regressionObject = regressionObject    # Regression class object - must be initialized with model!
        self.X = X                                  # Design matrix
        self.y = y                                  # Real data
        self.BETA = None                            # Matrix to contain all beta arrays computed in folds

    def bootstrapResampling(self, k, test_size):
        """
        Performs the bootstrap  resampling algorithm k times
        Uses the regression object from initialization
        Assumes regression already done

        Parameters0
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
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size)
        N = X_train.shape[0]
        N_betas = np.shape(self.X)[1]

        # containers for computed quantities
        yPredictions = np.empty((y_test.shape[0], k))
        y_trainPredictions = np.empty((y_train.shape[0], k))
        y_bootstrap= np.empty((y_train.shape[0], k))
        self.BETA = np.zeros((N_betas, k))
        r2 = np.zeros(k)

        #resampling loop
        for i in range(k):

            #Fit model to random indices
            idx = np.random.randint(0, N, N)
            X_idx, y_idx = X_train[idx], y_train[idx]
            self.regressionObject.computeRegression(X_idx, y_idx)

            # Make prediction and save data 
            yPredictions[:, i] = self.regressionObject.returnPrediction(X_test)
            y_trainPredictions[:, i] = self.regressionObject.returnPrediction(X_idx)
            y_bootstrap[:, i] = y_idx
            self.BETA[:, i] = self.regressionObject.beta
            r2[i] = dataFunctions.r2(yPredictions[:, i], y_test) 
        
        # error scores set as class params
        self.mse = dataFunctions.mse(yPredictions, y_test)
        self.mseTrain = dataFunctions.mse(y_trainPredictions, y_bootstrap)
        self.r2 = np.mean(r2) 
        self.yBias = dataFunctions.bias(y_test, yPredictions) 
        self.yPrecitionsVariance = dataFunctions.variance(yPredictions) 
        self.betaVariance = np.var(self.BETA, axis = 1)
        self.betaMean = np.mean(self.BETA, axis = 1)

        print('Bootstrap complete. Stat set as class variables.')

    def returnStats(self):
        return self.mse, self.mseTrain, self.r2, self.yBias, self.yPredictionsVariance
    
    def returnBetas(self):
        return self.betaMean, self.betaVariance
