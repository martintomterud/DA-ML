# ----- PACKAGE IMPORT ----- #

#libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"

import seaborn as sns
from sklearn.datasets import load_breast_cancer

# src
from src import neuralNet, prepareData

np.random.seed(seed=int(42)) 

# ---- END PACKAGE IMPORT ---- #


def neural_network_classification():
    """
    Use the neural network to explore parameter space
    in learing rates and l2 penalties and
    plot confusion matrix
    """
    # Import breast cancer data
    X, y = load_breast_cancer(return_X_y=True)
    

    #Define test parameters
    lambdas = np.logspace(-5, 1, 7)
    lambda_labels = np.linspace(-5, 1, 7)
    learningRates = np.logspace(-6, 0, 7)
    lr_labels = np.linspace(-6,0, 7)

    #Storage container
    accuracies = np.zeros([len(learningRates), len(lambdas)])

    max_iter = 1000
    # Averaging result over 'num_avg' runs
    num_avg = 5

    for i in range(len(learningRates)):
        for j in range(len(lambdas)):
            for _ in range(num_avg):
                X_train, X_test, y_train, y_test = (
                    prepareData.prepare_classification(X, y)
                )

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
    fig.savefig("figures/logreg.pdf", bbox_inches = 'tight')
