import numpy as np

def franke(x, y):
    """
    Returns the franke function of the variables (x, y), defined on [0,1]x[0,1]
    Parameters
    ---------------
    x : numpy.array
        Meshgrid for vectorized evaluation
    y : numpy.array
        Meshgrid
    Returns
    ---------------
    numpy.array
        Franke function at specified (x,y) input 
    """

    return (
        (3/4) * np.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4) 
        + (3/4) * np.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10) 
        + (1/2)  * np.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4) 
        - (1/5)  * np.exp(-((9*x - 4)**2) - ((9*y - 7)**2)) 
    )