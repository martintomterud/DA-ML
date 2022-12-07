import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.experimental import numpy as tnp

from pde_model_hm import PDEModel, ConditionLayer


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

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=6e-1)

    model = create_model(
        func=initial_condition,
        hidden_layers=hidden_layers,
        activation=activation,
        initializer=initializer
    )
    model.compile(optimizer=optimizer)

    N_train = 300
    N_test = 100
    epochs = 1000

    T = .5
    x_train = rng.random((N_train, 1), dtype="float32")
    t_train = T * rng.random((N_train, 1), dtype="float32")

    model.fit([x_train, t_train], epochs=epochs)

    x_test = rng.random((N_test, 1), dtype="float32")
    t_test = T * rng.random((N_test, 1), dtype="float32")

    f_pred = model.predict([x_test, t_test])

    f_anal = f_analytic(x_test, t_test)

    fig, ax = plt.subplots()
    ax.scatter(x_test, t_test, c=f_pred, cmap="Reds")
    fig.savefig("scatter_pde.pdf")

    fig2, ax2 = plt.subplots()
    ax2.scatter(x_test, t_test, c=f_anal, cmap="Reds")
    fig2.savefig("scatter_analytic.pdf")


main()
