import numpy as np 


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


