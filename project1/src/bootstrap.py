import numpy as np

import regressionClass
import designMatrix

class Bootstrap:

    """
    Bootstrap resampling of regression class
    """

    def __init__(self, regressionObject, X):
        self.regressionObject = regressionObject    # Regression class object
        self.X = X                                  # Design matrix
        self.BETA = None                            # Matrix to contain all beta arrays computed in folds

    def bootstrapResampling(self, x, y, k):
        """
        Performs the bootstrap  resampling algorithm k times
        Uses the regression object from initialization
        Assumes regression already done

        Parameters
        ---------------
        x : np.array
            data set, 1d array
        y : numpy.array
            y data, 1d array
        k : int
            number of "folds", i.e. how many times we perform the algorithm
        
        Algorithm follows the bootstrap algorithm on this link:
        https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
        but loops over all k fold values to compute beta statistics
        
        """


        # Set BETA to correct size
        self.BETA = np.zeros((len(self.regressionObject.beta), k))

        #does not work, just copy pasted below from
        #link to adapt

        for i in range(k):
            t = np.zeros(datapoints)
            n = len(data)
            # non-parametric bootstrap         
            for i in range(datapoints):
                t[i] = np.mean(data[np.random.randint(0,n,n)])


