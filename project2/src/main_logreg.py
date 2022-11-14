# ----- PACKAGE IMPORT ----- #

#libraries
from math import degrees
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time 
import os
import pandas
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#scikit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml


#scipy
from scipy.special import softmax

#src
import franke
import dataFunctions
import prepareData
import sgd
import logisticRegression
#set seed for numpy random using time 
#np.random.seed(seed=int(time.time())) 
#set const seed for debugging
np.random.seed(seed=int(42)) 

# ---- END PACKAGE IMPORT ---- #


"""
This file contains functions that are executed to answer
parts e) of project 2

"""

def main_logreg():
    """
    Uses scikit learn functionality from
    https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html#sphx-glr-auto-examples-semi-supervised-plot-self-training-varying-threshold-py
    to correctly handle importing the breast cancer data
    """
    # Import breast cancer data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Ntot = len(y)
    #Preprocess data: scale, shuffle and so on
    X, y = shuffle(X, y, random_state=100)
    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    print(y_test)
    #Logistic regression parameters
    learningRates = np.logspace(-5, 0, 6)
    totEpochs = 1000
    epochsPerIteration = 10
    numIterations = int(totEpochs/epochsPerIteration)
    momentum = 1e-1
    N = X_train.shape[0]
    Nweights = X_train.shape[1]
    #use batch sizes that makes sense with memory (i.e. multiples of 8 )
    numBatches = [int(N / 64), int(N/128)]

    #storage container for regression scores
    regressionScores = np.zeros([2, len(learningRates), numIterations])
    
    #initial random weights
    initialWeights = np.random.randn(Nweights)
    print(initialWeights[0])
    print(X_test)
    #start loop for logreg
    for i in range(len(numBatches)):
        for j in range(len(learningRates)):
            regressionWeights = logisticRegression.logisticRegression(
                X_train,
                y_train,
                initialWeights,
                numBatches[i],
                epochsPerIteration,
                learningRates[j],
                momentum,
                #lamb = 0.1
            )
            prediction = softmax(X_test @ regressionWeights)
            score = accuracy_score(y_test, prediction)
            regressionScores[i, j, 0] = score
            for k in range(1, numIterations):
                regressionWeights = logisticRegression.logisticRegression(
                    X_train,
                    y_train,
                    regressionWeights,
                    numBatches[i],
                    epochsPerIteration,
                    learningRates[j],
                    momentum,
                    #lamb = 0.1
                )
                prediction = softmax(X_test @ regressionWeights)
                score = accuracy_score(y_test, prediction)
                regressionScores[i, j, k] = score
                #print(prediction)
    
    plt.figure()
    x_axis = np.linspace(1, totEpochs, numIterations)
    for i in range(len(learningRates)):
        plt.plot(x_axis, regressionScores[0, i, :])
    plt.show()

main_logreg()
