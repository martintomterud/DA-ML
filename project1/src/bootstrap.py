import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import regressionClass
import designMatrix

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

        Parameters
        ---------------
        k : int
            number of "folds", i.e. how many times we perform the algorithm
        test_size : float
            size of test data for use in scikit learn train_test_split
        
        Algorithm follows the bootstrap algorithm on this link:
        https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
        but loops over all k fold values to compute beta statistics
        
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
            r2[i] = r2_score(yPredictions[:, i], y_test)
        
        # error scores set as class params
        self.mean_mse = np.mean(np.mean((yPredictions - y_test)**2), axis = 1, keepdims = True)
        self.mean_mse_train =  np.mean(np.mean((y_trainPredictions - y_bootstrap)**2), axis = 1, keepdims = True)
        self.yBias = np.mean( (y_test - np.mean(yPredictions, axis=1, keepdims=True))**2)
        self.yPrecitionsVariance = np.mean(np.var(yPredictions, axis=1, keepdims=True))
        self.betaVariance = np.var(self.BETA, axis = 1)
        self.betaMean = np.mean(self.BETA, axis = 1)

        print('Bootstrap complete. Stat set as class variables.')

