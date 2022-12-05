import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.keras.initializers import GlorotNormal, RandomNormal

class ODEModel(tf.keras.Model):
    # Keras model with custom training step
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track loss during training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            # Forward pass
            func = self(x, training=True)
            # Compute derivative of 'func'
            df_dx = tape.gradient(func, x)
            # Compute costum loss
            loss = ode_loss(func, df_dx)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # List of `Metric` objects
        return [self.loss_tracker]


def create_model(hidden_layers=[10,], u_0=10., initializer=GlorotNormal()):
    x = tf.keras.Input(shape=(1))
    nn = x

    for n in hidden_layers:
        nn = tf.keras.layers.Dense(
            units=n,
            activation=tf.keras.activations.sigmoid,
            kernel_initializer=initializer,
            bias_initializer=initializer
        )(nn)

    nn = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=initializer,
        bias_regularizer=initializer
    )(nn)

    u = ConditionLayer(x, u_0)(nn)

    return ODEModel(inputs=x, outputs=u)



class ConditionLayer(tf.keras.layers.Layer):
    # Layer which forces initial conditions on the network
    def __init__(self, x, u_0=0.):
        super(ConditionLayer, self).__init__()
        self.x = x
        self.u_0 = u_0

    def __call__(self, nn):
        # Function defining the initial conditions
        return self.u_0 + self.x * nn


def ode_loss(u, du_dx):
    eq = du_dx + 2. * u
    sq = tf.math.square(eq)
    return tf.math.reduce_mean(sq)


def u_analytic(x, u_0=10., gamma=2.):
    return u_0 * tnp.exp(-gamma * x)


def func1(u_0):
    return lambda x: u_0

def func2():
    return lambda x: x

def main():
    rng = np.random.default_rng()

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    u_0 = 10.
    hidden_layers = [10, 5]

    model = create_model(u_0=u_0)
    model.compile(optimizer=optimizer)

    N = 10
    x_train = rng.random((N, 1), dtype="float32")

    model.fit(x_train, epochs=2000)

    x_test = rng.random((100, 1), dtype="float32")
    u_pred = model.predict(x_test)

    u_anal = u_analytic(x_test, u_0)

    fig, ax = plt.subplots()
    ax.plot(x_test, u_pred, '.')
    ax.plot(x_test, u_anal, '.')
    fig.savefig("second.pdf")


main()
