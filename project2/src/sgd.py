import numpy as np

def SGD(x, y, startWeights, numBatches, numEpochs, learningRate, costFunction, *lamb):
    """
    SGD method 
    
    Algorithm based on lec notes from week 39, link
    https://compphysics.github.io/MachineLearning/doc/pub/week39/html/week39.html

    Parameters
    -------------------
    x :             ndarray - Design matrix
    y :             1d array - Response / Input vector of model
    startWeights :       1d array - Start weights
    numBatches :    int - Number of batches to split permuted arrays into
    numEpochs :     int - Number of epochs
    learningRate :  float - The learning rate
    costFunction :  function - pass to the fuction used to compute the gradient
    *lamb :         float - optional of costFunctino requires lambda parameter

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            gradient = costFunction(x[splitIndeces[b]], y[splitIndeces[b]], weights, *lamb) / x[splitIndeces[b]].shape[0]
            #change weights according to learning rate
            weights -= learningRate * gradient
    return weights

def sgdm(x, y, startWeights, numBatches, numEpochs, learningRate, momentum, costFunction, *lamb):
    """
    SGDM method 
    Stoch gradient descent with momentum

    Algorithm based on lec notes from week 39, link
    https://compphysics.github.io/MachineLearning/doc/pub/week39/html/week39.html

    Only differs in the way the new weights are computed

    Parameters
    -------------------
    x :             ndarray - Design matrix
    y :             1d array - Response / Input vector of model
    startWeights :       1d array - Start weights
    numBatches :    int - Number of batches to split permuted arrays into
    numEpochs :     int - Number of epochs
    learningRate :  float - The learning rate
    momentum :      float - The momentum in [0, 1]
    costFunction :  function - pass to the fuction used to compute the gradient
    *lamb :         float - optional of costFunctino requires lambda parameter

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights
    dweight = 0 #starts at 0 and is incremented

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            gradient = costFunction(x[splitIndeces[b]], y[splitIndeces[b]], weights, *lamb) / x[splitIndeces[b]].shape[0]
            #change weights according to learning rate
            dweight = dweight * momentum - learningRate*gradient
            weights += dweight
    return weights

def adaptiveGradient(x, y, startWeights, numBatches, numEpochs, learningRate, costFunction, *lamb):
    """
    Adaptiv gradient method
   
    Algorithm based on lec notes from week 39, link
    https://compphysics.github.io/MachineLearning/doc/pub/week39/html/week39.html

    Parameters
    -------------------
    x :             ndarray - Design matrix
    y :             1d array - Response / Input vector of model
    startWeights :       1d array - Start weights
    numBatches :    int - Number of batches to split permuted arrays into
    numEpochs :     int - Number of epochs
    learningRate :  float - The learning rate
    costFunction :  function - pass to the fuction used to compute the gradient
    *lamb :         float - optional of costFunctino requires lambda parameter

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights
    gradientStore = np.zeros(len(startWeights))

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            gradient = costFunction(x[splitIndeces[b]], y[splitIndeces[b]], weights, *lamb) / x[splitIndeces[b]].shape[0]
            #store square gradient
            gradientStore += np.dot(gradient, gradient)
            #change weights according to learning rate 
            weights -= learningRate * gradient / np.sqrt(gradientStore)
    return weights

def RMSporp(x, y, startWeights, numBatches, numEpochs, learningRate, discountFactor, costFunction, *lamb):
    """
    RMS porp
   
    Algorithm based on lec notes from week 39, link
    https://compphysics.github.io/MachineLearning/doc/pub/week39/html/week39.html

    Parameters
    -------------------
    x :                 ndarray - Design matrix
    y :                 1d array - Response / Input vector of model
    startWeights :      1d array - Start weights
    numBatches :        int - Number of batches to split permuted arrays into
    numEpochs :         int - Number of epochs
    learningRate :      float - The learning rate
    discountFactor :    float - The discount factor; deafult suggest at 0.9
    costFunction :      function - pass to the fuction used to compute the gradient
    *lamb :             float - optional of costFunctino requires lambda parameter

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights
    storedScaling = 0

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            gradient = costFunction(x[splitIndeces[b]], y[splitIndeces[b]], weights, *lamb) / x[splitIndeces[b]].shape[0]
            #store square gradient
            storedScaling = discountFactor * storedScaling + (1 - discountFactor) * np.dot(gradient, gradient)
            #change weights according to learning rate 
            weights -= learningRate * gradient / np.sqrt(storedScaling)
    return weights

def ADAM(x, y, startWeights, numBatches, numEpochs, learningRate, firstMomentFactor, secondMomentFactor, costFunction, *lamb):
    """
    RMS porp
   
    Algorithm based on lec notes from week 39, link
    https://compphysics.github.io/MachineLearning/doc/pub/week39/html/week39.html

    Parameters
    -------------------
    x :                 ndarray - Design matrix
    y :                 1d array - Response / Input vector of model
    startWeights :      1d array - Start weights
    numBatches :        int - Number of batches to split permuted arrays into
    numEpochs :         int - Number of epochs
    learningRate :      float - The learning rate
    firstMomentFactor :    float - The discount factor of first moment; deafult suggest at 0.9
    secondMomentFactor :    float - The discount factor of second moment; deafult suggest at 0.9
    costFunction :      function - pass to the fuction used to compute the gradient
    *lamb :             float - optional of costFunctino requires lambda parameter

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights
    firstMoment = 0
    secondMoment = 0

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            gradient = costFunction(x[splitIndeces[b]], y[splitIndeces[b]], weights, *lamb) / x[splitIndeces[b]].shape[0]
            #calculate first and second moments
            firstMoment = firstMomentFactor * firstMoment + (1 - firstMomentFactor)*gradient
            secondMoment = secondMomentFactor * secondMoment + (1 - secondMomentFactor)*np.dot(gradient, gradient)
            #bias corrected first and second moment
            firstMoment = firstMoment / (1 - firstMomentFactor)
            secondMoment = secondMoment / (1 - secondMomentFactor)
            #change weights according to learning rate 
            weights -= learningRate * firstMoment / np.sqrt(secondMoment)
    return weights

#########################################
# Typing up some cost functions as well #
#   These are equivalent to project 1   #
#########################################

def cost_OLS(X, y, prediction):
    return (y - X @ prediction).T @ (y - X @ prediction)

def cost_OLS_derivative(X, y, prediction):
    return -2 * X.T @ (y - X @ prediction)

def cost_Ridge(X, y, prediction, lambd):
    return (y - X @ prediction).T @ (y - X @ prediction) + lambd * prediction.T @ prediction

def cost_Ridge_derivative(X, y, prediction, lambd): 
    return -2 * X.T @ (y - X @ prediction) + 2 * lambd * prediction     
