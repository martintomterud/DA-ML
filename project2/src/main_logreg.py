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
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression


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

def sklearn_test():
    """
    Check what the accuracy should be using sklearns methods
    """
    #    Import breast cancer data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Ntot = len(y)
    #Preprocess data: scale, shuffle and so on
    X, y = shuffle(X, y, random_state=100)
    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Logreg using scikit
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    train_accuracy = logreg.score(X_test, y_test)

    #Scale on train data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Fit scaled data
    logreg.fit(X_train_scaled, y_train)
    train_accuracy_scaled = logreg.score(X_test_scaled, y_test)

    print(train_accuracy, train_accuracy_scaled)

sklearn_test()

def logreg_confmat():
    """
    Logistic regression to explore parameter space
    in learing rates and l2 penalties and
    plot confusion matrix
    """
    # Import breast cancer data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Ntot = len(y)
    #Preprocess data: scale, shuffle and so on
    X, y = shuffle(X, y, random_state=100)
    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Scale on train data as this gave better result from testing sklearn methods
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Define test parameters
    lambdas = np.logspace(-5, 0, 6)
    learningRates = np.logspace(-6, -1, 6)

    #Set constants for run
    numMiniBatch = 128
    numEpochs = 10
    momentum = 0.7

    #Storage container
    accuracies = np.zeros([len(learningRates), len(lambdas)])

    #Initial regression weights
    startWeights = np.zeros(X_train.shape[1])
    startWeights = np.random.randn(X_train.shape[1])
    #Loop over learning rates and lambdas
    for i in range(len(learningRates)):
        for j in range(len(lambdas)):
            regWeights = logisticRegression.logisticRegression(
                X_train_scaled,
                y_train,
                startWeights,
                numMiniBatch,
                numEpochs,
                learningRates[i],
                momentum,
                lamb = lambdas[j]
            )
            arg = X_test_scaled @ regWeights
            prediction = softmax(arg)
            prediction = logisticRegression.prediction(X_test_scaled, regWeights)
            classification = logisticRegression.classify(prediction)
            score = accuracy_score(y_test, classification)
            accuracies[i, j] = score
    
    print(accuracies)

logreg_confmat()

