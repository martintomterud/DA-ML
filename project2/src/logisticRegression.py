
import numpy as np
from scipy.special import softmax

def softMax(X, w):
    """
    Returns multiclass soft max function
    """
    return np.exp(X@w)/np.sum(np.exp(X@w), axis = 0, keepdims=True)

def softMaxCost(X, Y, P):
    """
    Returns the soft max cost function
    """
    return  X.T @ (P - Y) 

def sigmoid(x):
    """
    Returns sigmoid function
    """
    return 1.0 / (1 + np.exp(-x))

def prediction(X, w):
    """
    Uses sigmoid to compute a prediction from design matrix and weights
    """
    arg = np.dot(X, w)
    return sigmoid(arg)

def costFunction(X, y, w):
    """
    Alternate implementation of cost function
    """
    N = len(y)
    preds = prediction(X, w)
    cost = np.zeros(N)
    for i in range(N):
        if preds[i] >= 0.5:
            #Error term for y = 1
            cost[i] = -1 * y[i] * np.log(preds[i])
        else:
            #Error term for y = 0
            cost[i] = -1*(1 - y[i])* np.log(1 - preds[i])
    #Total
    #cost = true_cost - false_cost
    #Compute average
    avg_cost = cost.sum()/N
    return avg_cost

def classify(prediction):
    classvec = np.zeros(len(prediction))
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            classvec[i] = 1
    return classvec

def newWeight(X, y, w, learninRate, momentum):
    N = len(y)
    preds = prediction(X, w)
    gradient = np.dot(X.T, preds - y)
    gradient = gradient /N * learninRate
    w = w - gradient
    return w

def logisticRegression(x, y, startWeights, numBatches, numEpochs, learningRate, momentum, lamb = 0):


    """
    Logistic Regression for multiclass soft max function
    with momentum stochastic gradient descent, copy pasted from sgd.py
    and updated with the cost function of soft max.

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
    lamb = 0:         float - optional, add a float to include l2 regularization

    Returns
    --------------------
    weights : array - the best weights flund by the costFunction
    
    """
    #Initialize out weights
    weights = startWeights
    dweight = np.zeros(weights.shape) #starts at 0 and is incremented

    #loop over number of epochs
    for nE in range(numEpochs):
        #split array into batches
        permutationIndeces = np.random.permutation(len(x))
        splitIndeces = np.array_split(permutationIndeces, numBatches)
        #loop over batches
        for nB in range(numBatches):
            #pick out random batch and compute gradient
            b = np.random.randint(numBatches)
            #set up matrices 
            X = x[splitIndeces[b]]
            Y = y[splitIndeces[b]]

            #see theory section in report for derivation of the soft max and its derivative
            #probability = softMax(X, weights)
            probability = softmax(X @ weights)
            gradient = softMaxCost(X, Y, probability)
            #print(gradient)
            
            #adds l2 reg term if it is non-zero
            if lamb:
                gradient += lamb*weights 
            #Update weights
            dweight = dweight * momentum - learningRate*gradient / X.shape[0]
            weights = dweight + weights
    return weights

