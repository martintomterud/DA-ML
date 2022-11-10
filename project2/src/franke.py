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
    term_1 = (3/4) * np.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4)
    term_2 = (3/4) * np.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10)
    term_3 = (1/2)  * np.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4) 
    term_4 =-1*(1/5)  * np.exp(-((9*x - 4)**2) - ((9*y - 7)**2)) 
    return (
        term_1 + term_2 + term_3 + term_4
    )

def noisy_franke(x, y, mu, sigma, N):
    """
    Returns the franke function of the variables (x, y), defined on [0,1]x[0,1]
    with additionally included noise from a normal distribution on N(0,1)
   
    Parameters
    ---------------
    x : numpy.array
        Meshgrid for vectorized evaluation
    y : numpy.array
        Meshgrid
    mu : float
        mean of the normal distribution to draw random points from
    sigma : float
            stddev of the normal distribution
    n : int
        output shape of random points
        must be equal to length of x and y arrays
    Returns
    ---------------
    numpy.array
        Franke function at specified (x,y) input with random noise
    """

    term_1 = (3/4) * np.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4)
    term_2 = (3/4) * np.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10)
    term_3 = (1/2)  * np.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4) 
    term_4 =-1*(1/5)  * np.exp(-((9*x - 4)**2) - ((9*y - 7)**2)) 
    random_term = np.random.normal(mu, sigma, x.shape)
    return (
        term_1 + term_2 + term_3 + term_4 + random_term
    )

def noise_scale_franke(x, y, scale, N):
    """
    Returns the franke function of the variables (x, y), defined on [0,1]x[0,1]
    with additionally included noise in a different version from above
   
    Parameters
    ---------------
    x : numpy.array
        Meshgrid for vectorized evaluation
    y : numpy.array
        Meshgrid
    mu : float
        mean of the normal distribution to draw random points from
    sigma : float
            stddev of the normal distribution
    n : int
        output shape of random points
        must be equal to length of x and y arrays
    Returns
    ---------------
    numpy.array
        Franke function at specified (x,y) input with random noise
    """

    term_1 = (3/4) * np.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4)
    term_2 = (3/4) * np.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10)
    term_3 = (1/2)  * np.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4) 
    term_4 =-1*(1/5)  * np.exp(-((9*x - 4)**2) - ((9*y - 7)**2)) 
    random_term = scale * np.random.normal(0, 1, N)
    return (
        term_1 + term_2 + term_3 + term_4 + random_term
    )
