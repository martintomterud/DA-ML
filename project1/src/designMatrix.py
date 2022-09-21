import numpy as np


class designMatrix:
    """
    Class definition of the design matrix object

    Deafult constructor initialises a matrix filled with ones (1) in the 
    correct shape as determined by the length of the data vectors (N)
    and the polynomial degree 

    The advanced constructor uses the meshgrid arrays x and y 
    to create a matrix on the form
    [
        1                           # deg 0
        x y                         # deg 1
        x^2 xy y^2                  # deg 2
        x^3 x^2y xy^2 y^3           # deg 3
        ...
        x^(n) x^(n-1)y ... y^(n)    #deg n     
    ]
    """

    def __init__(self, N, deg):
        """
        Deafult constructor
        Creating a matrix of 1es in shape given by size of data vector (N)
        and degrees
        """
        M = int((deg + 1)*(deg + 2) / 2)
        self.X = np.ones((N, M))
    
    def designMatrix(self, x, y, deg):
        """
        Constructor for design matrix

        Paramteres
        --------------
        x : ndarray
        y : ndarray
        deg : int
                degree of polynomials to include in design
        
        Returns
        -----------
        X : (N, M) sized ndarray
        """
        N = len(x)
        M = int((deg + 1)*(deg + 2) / 2)
        self.X = np.ones((N, M))

        for i in range(1, deg + 1):
            j = int(i*(i+2)/2)
            for k in range(i + 1):
                self.X[:, j + k] = x**(i - k) * y**(k)
    

