"""Finding eigenvectors of a symmetric matrix by using a neural network"""
import numpy as np
import matplotlib.pyplot as plt

import nn_eigval

def main():
    """Testing the neural network"""
    # Same seed for all to debug
    np.random.seed(42)

    # Set all constants
    matrix_size = 6
    hidden_neurons = 50
    num_layers = 4
    max_epoch = 400
    learning_rate = 0.001
    guess = 80
    delta_treshold = 1e-17
    verbose = True

    M = nn_eigval.random_symmetric_matrix(matrix_size)
    eig_vals, eig_vecs = np.linalg.eig(M)
    eig_vecs = eig_vecs.T
    tf_eig_val, tf_eig_vecs, max_iter_step = nn_eigval.nn_eigval(
        M,
        matrix_size,
        hidden_neurons,
        num_layers,
        max_epoch,
        learning_rate,
        guess,
        delta_treshold,
        verbose
    )

    print(tf_eig_val, eig_vals)
    print(tf_eig_vecs.shape)
    normalized_eigvecs = np.zeros((max_epoch, 1, matrix_size))
    for i in range(max_epoch):
        normalized_eigvecs[i, 0, :] = (tf_eig_vecs[i, 0, :] / np.linalg.norm(tf_eig_vecs[i, 0, :]))
    plt.figure()
    print((M @ tf_eig_vecs[-1].T))
    print(tf_eig_val * tf_eig_vecs[-1])
    print(eig_vals[0]*eig_vecs[0], M @ eig_vecs[0])
    t = np.linspace(0, max_iter_step+1, max_iter_step+1)
    for i in range(matrix_size):
        plt.plot(t, normalized_eigvecs[:, 0, i])

    # Choose which eigenvector (find the correct tone by comparing values, run time is short)
    # eig_vec = 0
    for i in range(matrix_size):
        plt.axhline(y = eig_vecs[0, i], ls = 'dashed', alpha = 0.7)


    plt.show()

if __name__ == "__main__":
    main()
