import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import neuralNet
import prepareData

def main():
    rng = np.random.default_rng()

    n = 1000
    function = "polynomial"

    noise = True
    mean_noise = 0.
    std_noise = .02

    layers = 5
    nodes = 10

    hidden_layers = []
    for _ in range(layers):
        hidden_layers.append(nodes)
    
    learning_rate = .5
    alpha = .001
    batch_size = 32
    max_iter = 1000
    tol = 1e-4
    leak = .1
    init_scale = 10.

    # param = {"alpha": np.concatenate((np.array([0.]), np.logspace(-10, 0, 11))), "learning_rate_init": np.linspace(.3, .5, 3)}
    # param = {"alpha": [0], "learning_rate_init": np.linspace(.3, .8, 3)}
    param = {"alpha": [0], "learning_rate_init": [.5]}

    x_tr, x_te, y_tr, y_te = prepareData.generate_data(n, function, noise, mean_noise, std_noise)

    homemade = neuralNet.FFNN(
        model="regression",
        activation="sigmoid",
        weight_dist="normal",
        hidden_layers=hidden_layers,
        alpha=alpha,
        leak=leak,
        init_scale=init_scale,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        rng=rng,
        n_iter_no_change=10,
        verbose=True
    )

    skl_model = neuralNet.MLPRegNormalInit(
        hidden_layer_sizes=hidden_layers,
        activation="logistic",
        solver="sgd",                   # Alternatives: "sgd", "adam", "lbfgs"
        alpha=alpha,                    # default: .0001
        batch_size=batch_size,          # default: "auto" = min(200, n)
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        tol=tol,                        # default: 1e-4
        momentum=0.,                    # default: .9
    )

    print(type(homemade))

    homemade.fit(x_tr, y_tr)
    skl_model.fit(x_tr, y_tr)

    y_pred = homemade.predict(x_te)
    y_val = skl_model.predict(x_te)

    fig, axs = plt. subplots(2, 1, figsize=(5,8))
    axs[0].plot(x_te, y_te, '.')
    axs[0].plot(x_te, y_pred, '.')
    axs[1].plot(x_te, y_te, '.')
    axs[1].plot(x_te, y_val, '.')
    fig.savefig("test_ppnn.pdf")

    print(homemade.r2_score(x_te, y_te))
    print(homemade.mse_score(x_te, y_te))

    return 0

main()