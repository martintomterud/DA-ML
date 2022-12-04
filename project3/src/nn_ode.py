import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import matplotlib.pyplot as plt


class ODEModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            du_dx = tape.gradient(u_pred, x)
            # print(type(du_dx))
            loss = ode_loss(u_pred, du_dx)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        # self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]


class FunctionLayer(tf.keras.layers.Layer):
    def __init__(self, x, u_0=0.):
        super(FunctionLayer, self).__init__()
        self.x = x
        self.u_0 = u_0

    def __call__(self, nn):
        return self.u_0 + self.x * nn



def create_model(u_0=10.):
    x = tf.keras.Input(shape=(1))

    nn = tf.keras.layers.Dense(10, activation=tf.keras.activations.sigmoid)(x)
    nn = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(nn)

    # h2 = tf.keras.layers.Multiply()([x, nn])

    # u = tf.keras.layers.Lambda(lambda x: u_0 + x)(h2)
    u = FunctionLayer(x, u_0)(nn)

    return ODEModel(inputs=x, outputs=u)


def ode_loss(u, du_dx):
    eq = du_dx + 2. * u
    sq = tf.math.square(eq)
    return tf.math.reduce_mean(sq)


def u_analytic(x, u_0=10., gamma=2.):
    return u_0 * tnp.exp(-gamma * x)


def main():
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model = create_model()
    # We don't passs a loss or metrics here.
    model.compile(optimizer=optimizer)

    # Just use `fit` as usual -- you can use callbacks, etc.
    N = 50
    x = tnp.linspace(0, 1., N, dtype="float32")

    model.fit(x, epochs=1000)

    u_pred = model.predict(x)

    u_anal = u_analytic(x)

    fig, ax = plt.subplots()
    ax.plot(x,u_pred)
    ax.plot(x,u_anal)
    fig.savefig("second.pdf")


main()
