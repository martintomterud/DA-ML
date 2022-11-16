
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

import neuralNet
import prepareData

def main():
    rng = np.random.default_rng()

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    x_tr, x_te, y_tr, y_te = prepareData.prepare_classification(X, y)

    layers = 5
    nodes = 10

    hidden_layers = []
    for _ in range(layers):
        hidden_layers.append(nodes)

    learning_rate = .5
    alpha = .001
    batch_size = 32
    max_iter = 200
    tol = 1e-4
    leak = .1
    init_scale = 10.

    homemade = neuralNet.FFNN(
        model="classification",
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
        verbose=True
    )

    skl_model = MLPClassifier(
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

    homemade.fit(x_tr, y_tr)
    skl_model.fit(x_tr, y_tr)

    y_pred = homemade.predict(x_te)

    print(homemade.accuracy(x_te, y_te))

    print(skl_model.score(x_te, y_te))

    return 0

main()