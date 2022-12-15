"""Solves a PDE with the use of a neural network"""
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import pde_solver
from pde_model import ConditionLayer, PDEModel

matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["text.usetex"] = True
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "serif"


def create_model(
    initial_conditions,
    hidden_layers=[10,],
    activation=tf.keras.activations.sigmoid,
    initializer=tf.keras.initializers.GlorotNormal(seed=42)
):
    """Creates neural network model to solve PDE

    Returns:
        A PDEModel object, which inherits from keras.Model
    """
    x = tf.keras.Input(shape=(1))
    t = tf.keras.Input(shape=(1))
    nn = tf.keras.layers.concatenate([x, t])
    for n in hidden_layers:
        nn = tf.keras.layers.Dense(
            units=n,
            activation=activation,
            kernel_initializer=initializer,
            bias_initializer=initializer
        )(nn)

    nn = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=initializer,
        bias_regularizer=initializer
    )(nn)

    f = ConditionLayer(x, t, initial_conditions)(nn)

    return PDEModel(ode_loss, inputs=[x, t], outputs=f)

def ode_loss(df_t, d2f_x):
    """Computes loss to be minimised by network

    The differential equation to be solved is described by 'eq'
    """
    eq = d2f_x - df_t
    return tf.math.reduce_mean(tf.math.square(eq))

def initial_condition(x, t, neural_network):
    """Restricts output of network to initial conditions"""
    return tf.sin(np.pi * x) * (1 + t * neural_network)

def f_analytic(x, t):
    """Gives the analytic solution of the PDE for testing"""
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def main(N_train):
    """Testing the neural network"""
    rng = np.random.default_rng()

    # Settings for neural network
    hidden_layers = [10,]
    activation = tf.keras.activations.tanh
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=.8)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=6e-1)

    model = create_model(
        initial_conditions=initial_condition,
        hidden_layers=hidden_layers,
        activation=activation,
        initializer=initializer
    )
    model.compile(optimizer=optimizer)

    # N_train = 10
    N_test = 100
    epochs = 400

    T = .5
    x_train = rng.random((N_train, 1), dtype="float32")
    t_train = T * rng.random((N_train, 1), dtype="float32")

    model.fit([x_train, t_train], epochs=epochs)

    x = np.linspace(0, 1., N_test+1, dtype="float32")
    t = np.linspace(0, .5, N_test+1, dtype="float32")
    x_mesh, t_mesh = np.meshgrid(x, t)
    x_test = x_mesh.reshape(-1, 1)
    t_test = t_mesh.reshape(-1, 1)

    f_pred = model.predict([x_test, t_test])
    f_mesh = f_pred.reshape(x_mesh.shape)

    f_anal = f_analytic(x_mesh, t_mesh)

    sqr_err = pde_solver.sqr_err(f_mesh, f_anal)
    # rel_err = pde_solver.rel_err(f_mesh, f_anal)

    fig, ax = plt.subplots(figsize=(8,8))
    pan1 = ax.pcolormesh(x_mesh, t_mesh, f_mesh, cmap='Reds', rasterized=True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    cbar = fig.colorbar(pan1, ax=ax)
    cbar.set_label(r'$u(x, t)$')
    fig.savefig("nn_pred" + str(N_train) + ".pdf", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.pcolormesh(x_mesh, t_mesh, sqr_err, cmap="Greys", rasterized=True)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    fig.colorbar(im, ax=ax, label = 'Square Error')
    fig.savefig("nn_sqr_err" + str(N_train) + ".pdf", bbox_inches='tight')

    return 0


# Execute main for the desired number of training points
if __name__ == "__main__":
    main(10)
    main(100)
