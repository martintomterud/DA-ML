import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import neuralNet
import prepareData

def test_param():
    rng = np.random.default_rng()

    # Number of runs for average
    num_avg = 1

    # Set parameters for test function
    n = 1000
    function = "polynomial"
    noise = True
    mean_noise = 0.
    std_noise = .3

    # layers
    # layers_list = np.arange(1, 4)
    layers = 1
    # nodes_list = np.arange(30, 70, 3)
    nodes = 50

    learning_rates = np.linspace(.4, 1, 7)
    # learning_rate = .8
    # alphas = np.logspace(-10, -1, 5)
    alpha = 0.

    # init_scales = np.linspace(6., 20., 6)
    init_scale = 10.

    # print(layers_list)
    # print(nodes_list)

    batch_sizes = np.arange(8, 75, 5)
    # batch_size = 32

    max_iter = 100
    tol = 1e-4
    leak = .1

    x, y = prepareData.generate_data(n, function, noise, mean_noise, std_noise)
    x_tr, x_te, y_tr, y_te = prepareData.prepare_regression(x, y)

    length = len(learning_rates)
    # length2 = len(init_scales)

    # mse_list = np.zeros((length, length2))
    # r2_list = np.zeros((length, length2))


    mse_list = np.zeros(length)
    r2_list = np.zeros(length)

    for i in range(length):
        # for j in range(len(init_scales)):
            mse = 0
            r2 = 0
            hidden_layers = []
            for _ in range(layers):
                hidden_layers.append(nodes)

            for _ in range(num_avg):
                homemade = neuralNet.FFNN(
                    model="regression",
                    activation="sigmoid",
                    weight_dist="normal",
                    hidden_layers=hidden_layers,
                    alpha=alpha,
                    learning_rate=learning_rates[i],
                    leak=leak,
                    init_scale=init_scale,
                    max_iter=max_iter,
                    tol=tol,
                    batch_size=batch_sizes[i],
                    rng=rng,
                    n_iter_no_change=10
                )

                homemade.fit(x_tr, y_tr)

                mse += homemade.mse_score(x_te, y_te)
                r2 += homemade.r2_score(x_te, y_te)

            # mse_list[i, j] = mse/num_avg
            # r2_list[i, j] = r2/num_avg
            mse_list[i] = mse/num_avg
            r2_list[i] = r2/num_avg

    fig, ax = plt.subplots()
    # sns.heatmap(mse_list, annot = True, cbar = True)
    ax.plot(batch_sizes, np.log(mse_list), '.')
    fig.savefig("batch_size.pdf")

    # axs[0].plot(np.log(alphas), mse_list, '.')
    # axs[1].plot(np.log(alphas), r2_list, '.')
    # fig.savefig("best_lr.pdf")

    return 0

def test_best():
    rng = np.random.default_rng()

    # Number of runs for average
    num_avg = 10

    # Set parameters for test function
    n = 1000
    function = "polynomial"
    noise = True
    mean_noise = 0.
    std_noise = .06

    hidden_layers = [50]
    learning_rate = .8
    alpha = 0.
    init_scale = 10.
    batch_size = 32
    max_iter = 100
    tol = 1e-4
    leak = .1

    x, y = prepareData.generate_data(n, function, noise=noise, mean_noise = mean_noise, std_noise=std_noise)

    mse = 0
    r2 = 0

    for _ in range(num_avg):
        x_tr, x_te, y_tr, y_te = prepareData.prepare_regression(x, y)

        model = neuralNet.FFNN(
            model="regression",
            activation="sigmoid",
            weight_dist="normal",
            hidden_layers=hidden_layers,
            alpha=alpha,
            learning_rate=learning_rate,
            leak=leak,
            init_scale=init_scale,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            rng=rng,
            n_iter_no_change=10
        )
        model.fit(x_tr, y_tr)

        mse += model.mse_score(x_te, y_te)
        r2 += model.r2_score(x_te, y_te)
    
    mse /= num_avg
    r2 /= num_avg

    y_pred = model.predict(x_te)

    fig, ax = plt.subplots()
    ax.plot(x_te, y_te, '.')
    ax.plot(x_te, y_pred, '.')
    fig.savefig("best_test.pdf")

    print(mse)
    print(r2)

    return 0


# test_param()
test_best()