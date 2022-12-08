import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.experimental import numpy as tnp

from pde_model_hm import PDEModel, ConditionLayer
import pde_solver


def create_model(
    func, 
    hidden_layers=[10,], 
    activation=tf.keras.activations.sigmoid,
    initializer=tf.keras.initializers.GlorotNormal(seed=42)
):
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

    f = ConditionLayer(x, t, func)(nn)

    return PDEModel(ode_loss, inputs=[x, t], outputs=f)

#  The following three functions describe the ODE to be solved
def ode_loss(df_t, d2f_x):
    # ODE to be minimized
    eq = d2f_x - df_t
    return tf.math.reduce_mean(tf.math.square(eq))

def initial_condition(x, t, neural_network):
    # This functions wraps the neural network and makes it consistent with the
    # initial conditions
    return tf.sin(tnp.pi * x) * (1 + t * neural_network)

def f_analytic(x, t):
#     # Gives the analytic solution to the ODE for comparison
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def main():
    rng = np.random.default_rng()

    # Settings for neural network
    hidden_layers = [10,]
    activation = tf.keras.activations.tanh
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=.8)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=6e-1)

    model = create_model(
        func=initial_condition,
        hidden_layers=hidden_layers,
        activation=activation,
        initializer=initializer
    )
    model.compile(optimizer=optimizer)

    N_train = 200
    N_test = 100
    epochs = 400

    T = .5
    x_train = rng.random((N_train, 1), dtype="float32")
    t_train = T * rng.random((N_train, 1), dtype="float32")

    model.fit([x_train, t_train], epochs=epochs)

    x_test = rng.random((N_test, 1), dtype="float32")
    t_test = T * rng.random((N_test, 1), dtype="float32")

    f_pred = model.predict([x_test, t_test])

    f_anal = f_analytic(x_test, t_test)

    sqr_err = pde_solver.sqr_err(f_pred, f_anal)
    rel_err = pde_solver.rel_err(f_pred, f_anal)

    fig, ax = plt.subplots()
    im = ax.scatter(x_test, t_test, c=f_pred, cmap="Reds")
    fig.colorbar(im, ax=ax)
    fig.savefig("nn_pred.pdf")

    # fig2, ax2 = plt.subplots()
    # im2 = ax2.scatter(x_test, t_test, c=f_anal, cmap="Reds")
    # fig2.colorbar(im2, ax=ax2)
    # fig2.savefig("nn_analytic.pdf")

    fig, ax = plt.subplots()
    im = ax.scatter(x_test, t_test, c=sqr_err, cmap="Reds")
    fig.colorbar(im, ax=ax)
    fig.savefig("nn_sqr_err.pdf")

    fig, ax = plt.subplots()
    im = ax.scatter(x_test, t_test, c=rel_err, cmap="Reds")
    fig.colorbar(im, ax=ax)
    fig.savefig("nn_rel_err.pdf")

    return 0

main()
