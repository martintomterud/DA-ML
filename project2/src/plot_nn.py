import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import prepareData
import neuralNet

def main():
    plot_lr()
    return 0

def plot_lr():
    num_avg = 5

    # Set parameters for test data
    n = 1000
    function = "franke"
    noise = True
    mean_noise = 0.
    std_noise = .01

    # Generate test data
    x, y = prepareData.generate_data(n, function, noise, mean_noise, std_noise)
    x_tr, x_te, y_tr, y_te = prepareData.prepare_regression(x, y)

    # Parameters for model
    lr_log = np.logspace(-4, 0, 5)
    lr_lin = np.linspace(.1, 1., 5)
    max_iter = 500

    l_log = len(lr_log)
    l_lin = len(lr_lin)

    mse_test_log = np.zeros(l_log)
    mse_curve_log = np.zeros((l_log, max_iter-1))

    mse_test_lin = np.zeros(l_lin)
    mse_curve_lin = np.zeros((l_lin, max_iter-1))


    for i in range(l_log):
        mse = 0
        loss_curve = np.zeros(max_iter-1)
        for _ in range(num_avg):
            model = neuralNet.FFNN(
                learning_rate=lr_log[i],
                max_iter=max_iter
            )
            model.fit(x_tr, y_tr)

            mse += model.mse_score(x_te, y_te)
            loss_curve += model.loss_curve[1:]


        mse_curve_log[i] = loss_curve/num_avg
        mse_test_log[i] = mse/num_avg

    for i in range(l_lin):
        mse = 0
        loss_curve = np.zeros(max_iter-1)
        for _ in range(num_avg):
            model = neuralNet.FFNN(
                learning_rate=lr_lin[i],
                max_iter=max_iter
            )
            model.fit(x_tr, y_tr)

            mse += model.mse_score(x_te, y_te)
            loss_curve += model.loss_curve[1:]


        mse_curve_lin[i] = loss_curve/num_avg
        mse_test_lin[i] = mse/num_avg


    fig, ax = plt.subplots()
    for i in range(l_log):
        ax.plot(np.log10(mse_curve_log[i]), 
            label=r"\gamma = {0:1.1e}".format(lr_log[i])
        )

    fig.legend()
    fig.savefig("fig_H/lr_curve_log.pdf")

    fig2, ax2 = plt.subplots()
    
    ax2.plot(np.log10(lr_log), mse_test_log)
    fig2.savefig("fig_H/lr_mse_log.pdf")

    fig, ax = plt.subplots()
    for i in range(l_lin):
        ax.plot(np.log10(mse_curve_lin[i]), 
            label=r"\gamma = {0:1.1f}".format(lr_lin[i])
        )

    fig.legend()
    fig.savefig("fig_H/lr_curve_lin.pdf")

    fig2, ax2 = plt.subplots()
    
    ax2.plot(lr_lin, mse_test_lin)
    fig2.savefig("fig_H/lr_mse_lin.pdf")




    fig, ax  = plt.subplots(1, 2)
    fig.tight_layout()
    # #Ticks
    # #ax1.yaxis.set_major_locator(MultipleLocator(2000))
    # #ax1.yaxis.set_minor_locator(MultipleLocator(1000))
    # ax[0].xaxis.set_major_locator(MultipleLocator(1000))
    # ax[1].xaxis.set_major_locator(MultipleLocator(1000))
    # ax[0].xaxis.set_minor_locator(MultipleLocator(500))
    # ax[1].xaxis.set_minor_locator(MultipleLocator(500))
    # #ax[0].yaxis.set_major_locator(MultipleLocator(0.01))
    # #ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    # for a in ax:
    #     a.tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    #     a.tick_params(axis='x', top=True, direction = 'in', length = 10)
    #     a.tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    #     a.tick_params(axis='y', right=True, direction = 'in', length = 10)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('Epoch')
    # ax[1].set_ylabel('MSE')
    ax[1].set_xlabel('Epoch')

    x_axis = np.arange(1, max_iter)
    colors = ['r', 'b', 'g', 'darkorange', 'hotpink', 'teal', 'navy']
    for i in range(l_log):
        ax[0].semilogy(x_axis, mse_curve_log[i], color = colors[i], label = r'$\gamma = {0:1.0e}$'.format(lr_log[i])) 
    ax[0].legend(fontsize=12)

    for i in range(l_lin):
        ax[1].semilogy(x_axis, mse_curve_lin[i], color = colors[i], label = r'$\gamma = {0:1.2f}$'.format(lr_lin[i])) 

    ax[1].legend(fontsize=12)
    fig.savefig('fig_H/lr_test.pdf', bbox_inches = 'tight')

    return 0

main()