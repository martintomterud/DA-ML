"""Neural network model to solve a PDE

The training step of keras.Model is overriden to solve an ordinary differential
equation.
"""
import tensorflow as tf
from tensorflow import GradientTape

class PDEModel(tf.keras.Model):
    """Keras model with custom training step"""
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_loss = loss
        # Track loss during training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        unwrap = data[0]
        x = unwrap[0]
        t = unwrap[1]
        with GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            # Forward pass
            f = self(data, training=True)
            # f = self([x, t], training=True)
            # Compute derivative of 'f'
            df_x = tape.gradient(f, x)
            d2f_x = tape.gradient(df_x, x)
            df_t = tape.gradient(f, t)
            # Compute costum loss
            loss = self.ode_loss(df_t, d2f_x)

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


class ConditionLayer(tf.keras.layers.Layer):
    """Layer which forces initial conditions on the network"""
    def __init__(self, x, t, func):
        super(ConditionLayer, self).__init__()
        self.x = x
        self.t = t
        self.func = func

    def __call__(self, nn):
        # Function defining the initial conditions
        return self.func(self.x, self.t, nn)
