import tensorflow as tf
from tensorflow import GradientTape

class ODEModel(tf.keras.Model):
    # Keras model with custom training step
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_loss = loss
        # Track loss during training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, x):
        with GradientTape(persistent=True) as tape:
            tape.watch(x)
            # Forward pass
            func = self(x, training=True)
            # Compute derivative of 'func'
            df_dx = tape.gradient(func, x)
            # Compute costum loss
            loss = self.ode_loss(func, df_dx)

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
    # Layer which forces initial conditions on the network
    def __init__(self, x, func):
        super(ConditionLayer, self).__init__()
        self.x = x
        self.func = func

    def __call__(self, nn):
        # Function defining the initial conditions
        return self.func(self.x, nn)
