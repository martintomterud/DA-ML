import tensorflow as tf
from tensorflow import GradientTape

####################
#
#   I have just copy-pasted the ODE model and
#   altered the custom cost function call
#   so we can implement MSE between du_d2x and du_dt
#    as well as including a t tensor where only x tensor was before


class PDEModel(tf.keras.Model):
    # Keras model with custom training step
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_loss = loss
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
            du_dt = tape.gradient(du_dt, x)
            # Compute costum loss
            loss = self.ode_loss(func, du_d2x, du_dt)

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
    def __init__(self, x, t, func):
        super(ConditionLayer, self).__init__()
        self.x = x
        self.t = t
        self.func = func

    def __call__(self, nn):
        # Function defining the initial conditions
        return self.func(self.x, self.t, nn)
