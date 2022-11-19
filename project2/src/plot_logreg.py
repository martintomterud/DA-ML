# ----- PACKAGE IMPORT ----- #

#libraries
from math import degrees
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time 
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import seaborn as sns

#scikit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression


#scipy
from scipy.special import softmax

#src
import logisticRegression
import neuralNet
import prepareData
#set seed for numpy random using time 
#np.random.seed(seed=int(time.time())) 
#set const seed for debugging
np.random.seed(seed=int(42)) 

# ---- END PACKAGE IMPORT ---- #


def logreg_confmat(momentum):
    """
    Logistic regression to explore parameter space
    in learing rates and l2 penalties and
    plot confusion matrix
    """
    # Import breast cancer data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    # Ntot = len(y)
    # #Preprocess data: scale, shuffle and so on
    # X, y = shuffle(X, y, random_state=100)
    # #Split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # #Scale on train data as this gave better result from testing sklearn methods
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    

    #Define test parameters
    lambdas = np.logspace(-5, 1, 7)
    lambda_labels = np.linspace(-5, 1, 7)
    learningRates = np.logspace(-6, 0, 7)
    lr_labels = np.linspace(-6,0, 7)

    #Storage container
    accuracies = np.zeros([len(learningRates), len(lambdas)])

    max_iter = 1000
    num_avg = 5

    for i in range(len(learningRates)):
        for j in range(len(lambdas)):
            for _ in range(num_avg):
                X_train, X_test, y_train, y_test = prepareData.prepare_classification(X, y)

                model = neuralNet.FFNN(
                    model="classification",
                    learning_rate=learningRates[i],
                    alpha=lambdas[i],
                    max_iter=max_iter,
                )

                model.fit(X_train, y_train)

                accuracies[i,j] += model.accuracy(X_test, y_test)
            accuracies[i,j] /= num_avg

    fig, ax = plt.subplots()
    sns.heatmap(accuracies, annot = True, cbar = True)
    plt.xlabel(r'$\log_{10} \lambda$')
    plt.ylabel(r'$\log_{10} \gamma$')
    ax.set(xticklabels = np.round(lambda_labels, 0), yticklabels = np.round(lr_labels, 0))
    # ax.set_title(r'$\eta = $' +str(momentum))
    fig.savefig('fig_H/logreg.pdf', bbox_inches = 'tight')

logreg_confmat(0.)