import numpy as np
from scipy.linalg import pinv
import sklearn.linear_model as skl

class Regression:
    """
    Implementing three methods of linear regression as class

    Ordinary least squares (OLS)
        Solves linear regression by computing the analyticaly derived
        value fo beta from the project report

    Ridge
        Solves linear regression by computing the analyticaly derived
        value fo beta from the project report

    Lasso
        Solves linear regression by a call to the Lasso method
        in the sklearn.linear_model distribution
        
    """

    def __init__(self, method):
        """
        Initialises the class
        Sets all class variables to None

        Parameters
        -----------------
        method : string
                    The regression case to be executed
                    Available methods:
                        'ols', 'ridge', 'lasso'
        """
        self.method = method    #Least square method
        self.X = None           #Design matrix
        self.y = None           #Data
        self.yTilde = None      #Prediction
        self.beta = None        #Computed least square method
        self.lamb = 0.0         #Strength for use in ridge and lasso
        self.mse = None         #Mean square error
        self.r2 = None          #R2 score
        self.beta_var = None    #Variance of computed beta

    
    def getLamb(self):
        """
        Get function to print the lambda value of the regression
        """
        print('The current value of lambda is: ', self.lamb)
    
    def setLamb(self, newLamb):
        """
        Set function to set a new value of lambda for the regression class
        """
        self.lamb = newLamb
    
    def computeRegression(self, X, y):
        """
        ! Uses Python 3.10 functionality !  match - case

        Executes the regression specified by method

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
        """
        match self.method:
            case 'ols':
                self.computeOLSRegression(X, y)
            case 'ridge':
                self.computeRidgeRegression(X, y)
            case 'lasso':
                self.computeLassoRegression(X, y)
            case deafult:
                print('Regression method not recognized')


    def computeOLSRegression(self, X, y):
        """
        Execute to compute the linear regression

        Paramters
        -----------------
        X : numpy.array (N, M)
            design matrix. 
        
        y : numpy.array (N, 1)
            the input vector of the linear model
        """
        self.X = X
        self.y = y
        self.ols()

    def computeLassoRegression(self, X, y):
        """
        Execute to compute the lasso regression

        Paramters
        -----------------
        X : numpy.array (N, M)
            design matrix. 
        
        y : numpy.array (N, 1)
            the input vector of the linear model
        """
        self.X = X
        self.y = y
        self.lasso()

    def computeRidgeRegression(self, X, y):
        """
        Execute to compute the ridge regression

        Paramters
        -----------------
        X : numpy.array (N, M)
            design matrix. 
        
        y : numpy.array (N, 1)
            the input vector of the linear model
        """
        self.X = X
        self.y = y
        self.ridge()
    
    def ols(self):
        """
        Computes beta from the analytical solution of 
        the OLS method

        Formula documented in theory section of report

        In case of singular matrix, computes pseudo inverse using 
        scipy.linalg.pinv
        """
        X_T = self.X.T
        M_to_invert = X_T.dot(self.X)
        u, s, v = np.linalg.svd(M_to_invert)
        M_inv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        self.beta = M_inv.dot(X_T).dot(self.y)
        # self.beta = pinv(X_T.dot(self.X)).dot(X_T).dot(self.y)
    
    def ridge(self):
        """
        Computes beta from the analytical solution of 
        the ridge method

        Formula documented in theory section of report

        In case of singular matrix, computes pseudo inverse using 
        scipy.linalg.pinv
        """
        X_T = (self.X).T
        #find dimension p of matrix to generate correctly sized identity matrix
        M = np.shape(self.X)[1]
        Gamma = self.lamb * np.identity(M)
        Mat_to_invert = X_T.dot(self.X) + Gamma
        u, s, v = np.linalg.svd(Mat_to_invert)
        M_inv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        self.beta = M_inv.dot(X_T).dot(self.y)
        #self.beta = pinv(X_T.dot(self.X) + Gamma).dot(X_T).dot(self.y)

    def lasso(self):
        """
        Computes beta from the method implemented in 
        the sklearn library
        
        Leaving all input params open for easy adjustment

        Function documented at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso
        """

        clf = skl.Lasso(alpha=self.lamb, 
         fit_intercept=False,  
         precompute=False, 
         max_iter=int(1e3), 
         tol=0.001, 
         warm_start=False, 
         positive=False, 
         random_state=None, 
         selection='cyclic')

        clf.fit(self.X, self.y)

        self.beta = clf.coef_

    def makePrediction(self):
        """
        Make prediction using computed beta for general matrix
        X
        Sets the prediction as class variable

        Parameters
        -----------

        none, taken from self

        Computes
        -----------------
        
        yTilde : ndarray (N, 1)

        Prediction based on input data and computed beta. Defined in 
        equations in the report

        """
        if self.beta is None:
            print('Linear regression not performed \n')
            print('Exiting prediction')
            return 0
        self.yTilde = np.dot(self.X, self.beta)

    def returnPrediction(self, X):
        """
        Computes prediction from input matrix X
        For use in resampling techniques. 
        Does not set prediction, but returns predicted array.

        Parameters
        -----------

        X : ndarray, design matrix

        Computes
        -----------------
        
        yTilde : ndarray (N, 1)

        Prediction based on input data and computed beta. Defined in 
        equations in the report

        """
        if self.beta is None:
            print('Linear regression not performed \n')
            print('Exiting prediction')
            return 0
        return X @ self.beta


    
    def stat_mse(self):
        """
        Parameters
        -----------

        none, taken from self

        Computes
        -----------------
        Mean square error : float
            mse of prediction and y data
        """
        if self.yTilde is None:
            print('Prediction has not been computed \n')
            print('Exiting statistics')
            return 0
        N = self.y.size
        self.mse = np.sum((self.y - self.yTilde)**2) / N

    def stat_r2(self):
        """
        Parameters
        -----------

        none, taken from self

        Computes
        -----------------
        r2 score : float
            r2 score of prediction and y data
        """
        if self.yTilde is None:
            print('Prediction has not been computed \n')
            print('Exiting statistics')
            return 0
        N = self.y.size
        yMean = np.sum(self.y) / N
        self.r2 = 1 - np.sum((self.y - self.yTilde)**2) / np.sum((self.y - yMean)**2)        
