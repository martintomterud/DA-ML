from src import plot_logreg, plot_sgd, plot_nn, plot_nn_classifier

def main():

    plot_sgd.sgd_main(64)
    plot_sgd.sgd_main(128)
    plot_sgd.sgd_ridge(64)
    plot_sgd.adaptiveGradient(128)
    plot_sgd.tuneableRates(128)

    plot_logreg.logreg_confmat(0.0)
    plot_logreg.logreg_confmat(0.6)

    plot_nn.plot_lr()

    plot_nn_classifier.neural_network_classification()

    return 0

main()
