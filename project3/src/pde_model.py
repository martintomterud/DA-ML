import tensorflow as tf
from tensorflow import GradientTape
import numpy as np

####################
#
#   I have just copy-pasted the ODE model and
#   altered the custom cost function call
#   so we can implement MSE between du_d2x and du_dt
#    as well as including a t tensor where only x tensor was before

def prepare_arrays(grid_size, L, T):
    """
    Creates x and t arrays in ranges
    x in [0, L]
    t in [0, T]
    with grid_size random points
    """
    x = np.linspace(0, L, grid_size)
    t = np.linspace(0, T, grid_size)
    return x, t

def prepare_arrays_random(grid_size, L, T):
    """
    Creates x and t arrays in ranges
    x in [0, L]
    t in [0, T]
    with grid_size random points
    """
    x = np.random.ranf(0, L, grid_size)
    t = np.random.ranf(0, T, grid_size)
    # Make sure to include endpoints
    x[0] = 0
    x[-1] = L
    t[0] = 0
    t[-1] = T
    return x, t

class PDEModel(tf.keras.Model):
    """Keras model with custom training step"""
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_loss = loss
        # Track loss during training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")


    def train_step(self, x, t):
        with GradientTape(persistent=True) as tape:
            tape.watch(x)
            # Forward pass
            func = self(x, training=True)
            # Compute derivative of 'func'
            du_dx = tape.gradient(func, x)
            du_d2x = tape.gradient(du_dx, x)
            du_dt = tape.gradient(func, t)
            # Compute costum loss
            loss = self.pde_loss(du_d2x, du_dt)

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
