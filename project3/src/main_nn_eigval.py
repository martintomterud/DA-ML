"""Finding eigenvectors of a symmetric matrix by using a neural network"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"

import nn_eigval

# To choose correct eigenvalue
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main(guess):
    """Testing the neural network"""
    # Same seed for all to debug
    np.random.seed(42)

    # Set all constants
    matrix_size = 6
    hidden_neurons = 50
    num_layers = 4
    max_epoch = 400
    learning_rate = 0.001
    delta_treshold = 1e-17
    verbose = True

    M = nn_eigval.random_symmetric_matrix(matrix_size)
    eig_vals, eig_vecs = np.linalg.eig(M)
    eig_vecs = eig_vecs.T
    tf_eig_vals, tf_eig_vecs, max_iter_step = nn_eigval.nn_eigval(
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

    eigvalIDX = find_nearest(eig_vals, tf_eig_vals[-1])

    # Normalize eigenvectros given by nn
    normalized_eigvecs = np.zeros((max_epoch, 1, matrix_size))
    for i in range(max_epoch):
        normalized_eigvecs[i, 0, :] = (tf_eig_vecs[i, 0, :] / np.linalg.norm(tf_eig_vecs[i, 0, :]))
    plt.figure()

    #Checks
    print(tf_eig_vals[-1], eig_vals)
    print((M @ tf_eig_vecs[-1].T))
    print(tf_eig_vals[-1] * tf_eig_vecs[-1])
    print(eig_vals[0]*eig_vecs[0], M @ eig_vecs[0])

    # Compute mean square error
    mse = np.zeros(max_epoch)
    true_vec = eig_vals[eigvalIDX] * eig_vecs[eigvalIDX]
    for i in range(max_epoch):
        computed_vec = tf_eig_vals[i] * normalized_eigvecs[i, 0, :]
        sqr_error = (true_vec - computed_vec)**2
        mse[i] = np.mean(sqr_error)


    t = np.linspace(0, max_iter_step+1, max_iter_step+1)
    for i in range(matrix_size):
        plt.plot(t, normalized_eigvecs[:, 0, i])
    for i in range(matrix_size):
        plt.axhline(y = eig_vecs[eigvalIDX, i], ls = 'dashed', alpha = 0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Value of vector component')
    plt.title(r'$\lambda = $' + str(np.round(tf_eig_vals[-1], 2)))
    plt.savefig('NN_eigval_min.pdf', bbox_inches = 'tight')

    plt.figure()
    plt.plot(t, mse, color = 'r', alpha = 0.8, lw = 2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('NN_eigval_MSE_min.pdf', bbox_inches = 'tight')

    plt.show()

if __name__ == "__main__":
    main(-50)
    main(-50)
