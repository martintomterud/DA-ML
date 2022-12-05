import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.experimental import numpy as tnp

from ode_model import ODEModel, ConditionLayer


def create_model(
    func, 
    hidden_layers=[10,], 
    activation=tf.keras.activations.sigmoid,
    initializer=tf.keras.initializers.GlorotNormal(seed=42)
):
    x = tf.keras.Input(shape=(1))
    nn = x

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

    u = ConditionLayer(x, func)(nn)

    return ODEModel(ode_loss, inputs=x, outputs=u)

#  The following three functions describe the ODE to be solved
def ode_loss(f, df_dx):
    # ODE to be minimized
    eq = df_dx + 2. * f
    return tf.math.reduce_mean(tf.math.square(eq))

def initial_condition(x, neural_network):
    # This functions wraps the neural network and makes it consistent with the
    # initial conditions
    return 10. + x * neural_network

def u_analytic(x, u_0=10., gamma=2.):
    # Gives the analytic solution to the ODE for comparison
    return u_0 * tnp.exp(-gamma * x)


def main():
    rng = np.random.default_rng()

    # Settings for neural network
    hidden_layers = [10,]
    activation = tf.keras.activations.sigmoid
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=.4)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=6e-1)

    epochs = 300

    model = create_model(
        func=initial_condition,
        activation=activation,
        initializer=initializer
    )
    model.compile(optimizer=optimizer)

    N = 10
    # x_train = np.linspace(0., 1., N).reshape((N, 1))
    x_train = rng.random((N, 1), dtype="float32")

    model.fit(x_train, epochs=epochs)

    x_test = rng.random((100, 1), dtype="float32")
    u_pred = model.predict(x_test)

    u_anal = u_analytic(x_test)

    fig, ax = plt.subplots()
    ax.plot(x_test, u_pred, '.')
    ax.plot(x_test, u_anal, '.')
    fig.savefig("second.pdf")


main()
