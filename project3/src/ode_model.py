"""Neural network model to solve an ODE

The training step of keras.Model is overriden to solve an ordinary differential
equation.
"""
import tensorflow as tf
from tensorflow import GradientTape

class ODEModel(tf.keras.Model):
    """keras.Model with training step to solve an ODE"""
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_loss = loss
        # Track loss during training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        print(data.shape)
        with GradientTape(persistent=True) as tape:
            tape.watch(data)
            # Forward pass
            func = self(data, training=True)
            # Compute derivative of 'func'
            df = tape.gradient(func, data)
            # Compute costum loss
            loss = self.ode_loss(func, df)

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
    """Layer forces initial conditions on the network"""
    def __init__(self, x, initial_conditions):
        super(ConditionLayer, self).__init__()
        self.x = x
        self.initial_conditions = initial_conditions

    def __call__(self, nn, *args, **kwargs):
        # Function defining the initial conditions
        return self.initial_conditions(self.x, nn)
