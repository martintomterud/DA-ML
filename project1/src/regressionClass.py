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

    def __init__(self, lamb):
        """
        Initialises the class

        Paramterers
        -----------------

        lamb: numpy.float
            lambda paramter for use in ridge and lasso

        """
        self.lamb = lamb
    
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
    
    def computeRegression(self, method, X, y):
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
        match method:
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
        self.beta = pinv(X_T.dot(self.X)).dot(X_T).dot(self.y)
    
    def ridge(self):
        """
        Computes beta from the analytical solution of 
        the ridge method

        Formula documented in theory section of report

        In case of singular matrix, computes pseudo inverse using 
        scipy.linalg.pinv
        """
        X_T = self.X.T
        #find dimension p of matrix to generate correctly sized identity matrix
        M = np.shape(self.X)[1]
        Gamma = self.lamp * np.identity(M)
        self.beta = pinv(X_T.dot(self.X) + Gamma).dot(X_T).dot(self.y)

    def lasso(self):
        """
        Computes beta from the method implemented in 
        the sklearn library
        
        Leaving all input params open for easy adjustment

        Function documented at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso
        """

        clf = skl.Lasso(alpha=self.lamb, 
         fit_intercept=True, 
         normalize='deprecated', 
         precompute=False, 
         copy_X=True, 
         max_iter=1000, 
         tol=0.0001, 
         warm_start=False, 
         positive=False, 
         random_state=None, 
         selection='cyclic')

        clf.fit(self.X, self.y)

        self.beta = clf.coef_

    def makePrediction(self, X):
        """
        Make prediction using computed beta for general matrix
        X

        Parameters
        -----------

        X : ndarray (N, M)

        Returns
        -----------------
        
        tildeY : ndarray (N, 1)

        Prediction based oninput data and computed beta. Defined in 
        equations in the report

        """
        return np.dot(X, self.beta)
