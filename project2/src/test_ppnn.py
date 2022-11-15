import numpy as np
import matplotlib.pyplot as plt

import neuralNet
import prepareData

def main():
    rng = np.random.default_rng(42)

    n = 1000
    function = "polynomial"

    noise = True
    mean_noise = 0.
    std_noise = .02

    hidden_layers = [10,]

    # param = {"alpha": np.concatenate((np.array([0.]), np.logspace(-10, 0, 11))), "learning_rate_init": np.linspace(.3, .5, 3)}
    # param = {"alpha": [0], "learning_rate_init": np.linspace(.3, .8, 3)}
    param = {"alpha": [0], "learning_rate_init": [.5]}

    x_tr, x_te, y_tr, y_te = prepareData.generate_data(n, function, noise, mean_noise, std_noise)

    model = neuralNet.FFNN(
        max_iter=100, 
        batch_size=400,
        learning_rate=0.5,
        hidden_layers=hidden_layers,
        rng=rng
    )

    print(type(model))

    model.fit(x_tr, y_tr)

    fig, axs = plt. subplots(2, 1, figsize=(5,8))
    axs[0].plot(x_tr, y_tr, '.')
    plt.savefig("test_ppnn.pdf")

main()