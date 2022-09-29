import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import *
import matplotlib

class CrossValidation:
    """
    Class implementation of data sampling cross validation
    
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def splitTrainTest(self, method, X, y, test_size):
        """
        uses sklearn functionality to split data into testing and training
        performs linear regression on data and computes a prediction

        Parameters
        -----------------
        method : string
                    The regression case to be executed
                    Available methods:
                        'ols', 'ridge', 'lasso'

        X : numpy.array (N, M)
            design matrix. 
        
        y : numpy.array (N, 1)
            the input vector of the linear model
        
        test_size : float between 0 and 1
                    the size of the test data (usually about 0.2)
        """
        