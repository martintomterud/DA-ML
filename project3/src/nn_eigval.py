import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def random_symmetric_matrix(matrixSize):
    """
    Creates a random square matrix M of specified size
    Symmetrizes it by performing 
    M = (M + M^T)/2
    """
    M = np.random.rand(matrixSize, matrixSize)
    M =  ( M + np.transpose(M) ) / 2
    return M

def f_x(x, M, matrixSize):
    """
    Computes f(x) from tensorflow tensor x, matrix M
    """
    # x^TxM
    term1 = tf.tensordot(tf.transpose(x), x, axes = 1) * M
    # (1 - x^TMx)I
    term2 = (1 - tf.tensordot(tf.transpose(x), tf.tensordot(M, x, axes = 1),  axes = 1) ) * np.eye(matrixSize)
    sum = term1 + term2
    f = tf.tensordot(sum, x, axes = 1)
    return f

def nn_eigval(matrix, matrixSize, hidden_neurons, num_layers, max_epoch, learning_rate, guess, delta_treshold, verbose):
    
    """
    Neural network implementation that finds eigenvalues (and vectors) of 
    a given matrix by solving the differential equation


    dx/dt = - x(t) + f(x(t)),

    f(x) = [x^TxM + (1 - x^TMx)I]x

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
    delta_treshold   : float        - Parameter to break the loop if difference in eigenvalues is small
    verbose          : bool         - If true, prints status of calculation to console

    """

    # Step 1 : Trial solution
    I = np.identity(matrixSize) 
    x0 = np.random.rand(matrixSize)
    x0 = x0 / np.linalg.norm(x0) 

    # Step 2 : Convert to tensorflow
    x0 = np.reshape(x0, (1, matrixSize))
    x0_tf = tf.convert_to_tensor(x0, dtype='float64')
    I_tf = tf.convert_to_tensor(I)

    # Step 3 : Create neural network structure
    with tf.variable_scope('dnn'):
        prevLayer = x0_tf
        for i in range(num_layers):
            #set new current and swap to previous
            currLayer = tf.layers.dense(prevLayer, hidden_neurons, activation = tf.nn.sigmoid)
            prevLayer = currLayer
            #set ouput
            dnn_output = tf.layers.dense(prevLayer, matrixSize)
    
    
    # Step 4 : Loss function
    with tf.name_scope('loss'):
        
        #Trial solution
        x_trial = tf.transpose(dnn_output)
        f_trial = tf.transpose(f_x(x_trial, matrix, matrixSize))

        #Equilibrium point
        eigval_trial = tf.transpose(x_trial) @ matrix @ x_trial / (tf.transpose(x_trial) @ x_trial)
        
        x_trial = tf.transpose(x_trial)
        # Loss
        loss = tf.losses.mean_squared_error(f_trial, x_trial) + learning_rate * tf.losses.mean_squared_error([[guess]], eigval_trial)

    # Step 5 : Training
    with tf.name_scope('train'):
        #Choose optimizer and minimizer
        optimizer = tf.train.AdamOptimizer()
        training_operation = optimizer.minimize(loss)

    #Storing eigenvector as we progress
    eigVecContainer = []
    # Start session
    with tf.Session() as session:
        initial = tf.global_variables_initializer()
        initial.run()
        old_eigenVal = 0
        for i in range(max_epoch):
            session.run(training_operation)
            eigenVec = x_trial.eval()
            eigVecContainer.append(eigenVec)
            eigenVal = ((eigenVec @ (matrix @ eigenVec.T)) / (eigenVec @ eigenVec.T)) [0, 0]
            delta_eigVal = np.abs(old_eigenVal - eigenVal)
            old_eigenVal = eigenVal
            #PROGRESSBAR    
            if verbose:
                if i % 100 == 0:
                    print('Step: ', i,'/', max_epoch)
            
            #Break if delta smaller than treshold
            if delta_eigVal < delta_treshold:
                break
    
    return eigenVal, np.asarray(eigVecContainer), i

            

