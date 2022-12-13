"""Method to calculate eigenvector corresponding to the biggest eigenvalue,
for a symmetric matrix
"""
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def random_symmetric_matrix(matrix_size):
    """Creates a random square matrix of specified size"""
    M = np.random.rand(matrix_size, matrix_size)
    M =  (M + np.transpose(M)) / 2.
    return M

def f_x(x, M, matrix_size):
    """Computes f(x) from tensorflow tensor x, matrix M"""
    # x^TxM
    term1 = tf.tensordot(tf.transpose(x), x, axes = 1) * M
    # (1 - x^TMx)I
    prod = tf.tensordot(tf.transpose(x), tf.tensordot(M, x, axes=1), axes=1)
    term2 = (1 - prod) * np.eye(matrix_size)
    f = tf.tensordot(term1 + term2, x, axes = 1)
    return f

def nn_eigval(
    matrix,
    matrix_size,
    hidden_neurons,
    num_layers,
    max_epoch,
    learning_rate,
    guess,
    delta_treshold,
    verbose
):
    """Neural network implementation that finds eigenvalues (and vectors) of
    a given matrix by solving the differential equation

    dx/dt = - x(t) + f(x(t)),

    f(x) = [x^T x M + (1 - x^T M x) I] x

    where M is the matrix in question and I is the identity matrix.

    Input parameters
    --------------------------------------
    matrix           : n x n array  - The matrix to find the eigvals of
    matrixSize       : int          - The size of the matrix
    maxEpoch         : int          - Max number of iterations
    hidden_neurons   : int          - Number of hidden neurons in each layer
    num_layers       : int          - Number of layers between input and output
    learning_rate    : float        - LR for use in optimizer
    guess            : float        - Guess value of eigenval
    delta_treshold   : float        - Parameter to break the loop if difference
                                      in eigenvalues is small
    verbose          : bool         - If true, prints status of calculation to
                                      console
    """

    # Step 1: Trial solution
    # I = np.identity(matrix_size)
    x0 = np.random.rand(matrix_size)
    x0 = x0 / np.linalg.norm(x0)

    # Step 2: Convert to tensorflow
    x0 = np.reshape(x0, (1, matrix_size))
    x0_tf = tf.convert_to_tensor(x0, dtype='float64')
    # I_tf = tf.convert_to_tensor(I)

    # Step 3: Create neural network structure
    with tf.variable_scope('dnn'):
        prev_layer = x0_tf
        for i in range(num_layers):
            # Set new current and swap to previous
            curr_layer = tf.layers.dense(prev_layer, hidden_neurons, activation = tf.nn.sigmoid)
            prev_layer = curr_layer
            # Set ouput
            dnn_output = tf.layers.dense(prev_layer, matrix_size)

    # Step 4 : Loss function
    with tf.name_scope('loss'):

        # Trial solution
        x_trial = tf.transpose(dnn_output)
        f_trial = tf.transpose(f_x(x_trial, matrix, matrix_size))

        # Equilibrium point
        eigval_trial = tf.transpose(x_trial) @ matrix @ x_trial / (tf.transpose(x_trial) @ x_trial)

        x_trial = tf.transpose(x_trial)
        # Loss
        loss = (tf.losses.mean_squared_error(f_trial, x_trial)
                + learning_rate
                * tf.losses.mean_squared_error([[guess]], eigval_trial)
        )

    # Step 5 : Training
    with tf.name_scope('train'):
        # Choose optimizer and minimizer
        optimizer = tf.train.AdamOptimizer()
        training_operation = optimizer.minimize(loss)

    # Storing eigenvector as we progress
    eig_vec_container = []
    # Start session
    with tf.Session() as session:
        initial = tf.global_variables_initializer()
        initial.run()
        old_eigen_val = 0
        for i in range(max_epoch):
            session.run(training_operation)
            eigen_vec = x_trial.eval()
            eig_vec_container.append(eigen_vec)
            eigen_val = ((eigen_vec @ (matrix @ eigen_vec.T)) / (eigen_vec @ eigen_vec.T)) [0, 0]
            delta_eig_val = np.abs(old_eigen_val - eigen_val)
            old_eigen_val = eigen_val
            # Progress bar
            if verbose:
                if i % 100 == 0:
                    print('Step: ', i,'/', max_epoch)

            # if delta_eig_val < delta_treshold:
            #     break

    return eigen_val, np.asarray(eig_vec_container), i
