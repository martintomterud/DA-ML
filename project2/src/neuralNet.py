import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor


class FFNN:
    """Feedforward neural network"""

    # Constructor
    def __init__(
        self,
        model="regression",
        hidden_layers=[10,],
        activation="sigmoid",
        # solver="sgd",
        alpha=0.,
        leak=.01,
        learning_rate=.5,
        max_iter=200,
        tol=1e-4,
        # momentum=0.,
        batch_size=10,
        weight_dist="normal",
        init_scale=1.,
        rng=np.random.default_rng(),
        verbose=False,
        n_iter_no_change=10
    ):
        self.model = model
        self.hidden_layers = hidden_layers
        self.activation = activation
        # self.solver = solver
        self.alpha = alpha
        self.leak = leak
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        # self.momentum = momentum
        self.batch_size = batch_size
        self.weight_dist = weight_dist
        self.init_scale = init_scale
        self.rng = rng


        self.verbose = verbose
        self.n_iter_no_change = n_iter_no_change

        self.n_layers = len(hidden_layers) + 2
        self.out_activation = "identity"
        self.loss = "squared_error"
        self.layer_units = None
        self.weights = None
        self.intercepts = None

        self.fitted = False

    def _init_weights(self):
        # Initialise the weights and intercepts.
        self.weights = []
        self.intercepts = []

        for i in range(self.n_layers-1):
            dim_in = self.layer_units[i]
            dim_out = self.layer_units[i+1]

            # Scale as suggested by Glorot et. al. (2010) 
            # (see report for full citation)
            scale = self.init_scale * np.sqrt(2./(dim_in + dim_out))

            match self.weight_dist:
                case "normal":
                    self.weights.append(
                        self.rng.normal(0., scale, (dim_in, dim_out))
                    )
                case "uniform":
                    self.weights.append(
                        self.rng.uniform(-scale, scale, (dim_in, dim_out))
                    )

            self.intercepts.append(np.zeros(dim_out))

    def _feed_forward(self):
        hidden_act = ACTIVATION[self.activation]
        out_act = ACTIVATION[self.out_activation]

        for i in range(self.n_layers-1):
            self.activations[i+1] = (
                self.activations[i] @ self.weights[i] + self.intercepts[i]
            )

            if i != self.n_layers-2:
                if self.activation == "leaky_relu":
                    self.activations[i+1] = (
                        hidden_act(self.activations[i+1], self.leak)
                    )
                else:
                    self.activations[i+1] = hidden_act(self.activations[i+1])
        if self.out_activation == "leaky_relu":
            self.activations[self.n_layers-1] = (
                out_act(self.activations[self.n_layers-1], self.leak)
            )
        else:
            self.activations[self.n_layers-1] = (
                out_act(self.activations[self.n_layers-1])
            )

        return self.activations

    def _backprop(self, X, y):

        # Propegate forward
        self._feed_forward()

        # Propegate backwards
        last = self.n_layers - 2

        err = self.activations[-1] - y
        self.deltas[last] = err

        # Get loss
        loss = .5 * (err.T @ err)[0,0] / self.n_samples
        # loss = ((y - self.activations[-1]) ** 2).mean() / 2


        # Add L2 regularization term to loss
        values = 0
        for s in self.weights:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / self.n_samples


        # Compute gradient for the last layer
        self._compute_loss_grad(last)

        derivative = DERIVATIVES[self.activation]

        # Iterate over the hidden layers
        for i in range(last, 0, -1):
            self.deltas[i-1] = self.deltas[i] @ self.weights[i].T
            if self.activation == "leaky_relu":
                self.deltas[i-1] *= derivative(self.activations[i], self.leak)
            else:
                self.deltas[i-1] *= derivative(self.activations[i])
            self._compute_loss_grad(i - 1)

        return loss

    def _compute_loss_grad(self, layer):
        """Compute the gradient of loss with respect to weights and intercept 
        for specified layer. This function does backpropagation for the 
        specified layer.
        """
        self.weight_grads[layer] = (
            self.activations[layer].T @ self.deltas[layer]
        )
        # self.weight_grads[layer] += self.alpha * self.weights[layer]
        self.weight_grads[layer] /= self.n_samples

        self.intercept_grads[layer] = np.mean(self.deltas[layer], 0)

    def init_fit(self, X, y):

        self.n_samples, self.n_features = X.shape
        self.n_outputs_ = y.shape[1]

        # List of size of every layer
        self.layer_units = (
            [self.n_features] 
            + self.hidden_layers 
            + [self.n_outputs_]
        )

        # Initialise weights and intercepts
        self._init_weights()

        # Initialise variables for recording progress during fit
        self.loss_curve = []
        self.best_loss = np.inf

        # Initialize lists
        self.activations = [None] + [None] * (self.n_layers - 1)
        self.deltas = [None] * (self.n_layers - 1)

        self.weight_grads = []
        for i in range(self.n_layers-1):
            self.weight_grads.append(
                np.zeros(
                    (self.layer_units[i], self.layer_units[i+1]),
                    dtype=X.dtype
                )
            )

        self.intercept_grads = []
        for i in range(self.n_layers-1):
            self.intercept_grads.append(
                np.zeros(self.layer_units[i+1], dtype=X.dtype)
            )

    def fit(self, X, y,):

        self.init_fit(X,y)

        no_improvement_count = 0

        for it in range(self.max_iter):
            # Random permutation of indecies to avoid cycles
            sample_idx = self.rng.permutation(self.n_samples)

            num_batches = self.n_samples//self.batch_size
            if num_batches == 0:
                num_batches = 1

            batches = np.array_split(
                sample_idx, 
                num_batches
            )

            tot_loss = 0.0
            for batch_idx in batches:
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                self.activations[0] = X_batch

                batch_loss = self._backprop(
                    X_batch,
                    y_batch,
                )

                tot_loss += batch_loss * len(batch_idx)

                # Update weights and intercepts (SGD)
                for i in range(self.n_layers-1):
                    self.weights[i] -= (
                        self.learning_rate * self.weight_grads[i]
                    )
                    self.intercepts[i] -= (
                        self.learning_rate * self.intercept_grads[i]
                    )

            self.loss = tot_loss / X.shape[0]

            print(self.loss)

            self.loss_curve.append(self.loss)
            if self.verbose:
                print("Iteration %d, loss = %.8f" % (it, self.loss))

            # update no_improvement_count based on training loss or
            # validation score according to early_stopping
            if self.loss_curve[-1] > self.best_loss - self.tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if self.loss_curve[-1] < self.best_loss:
                self.best_loss = self.loss_curve[-1]


            if no_improvement_count > self.n_iter_no_change:
                # not better than last `n_iter_no_change` iterations by tol
                # stop or decrease learning rate
                msg = (
                    "Training loss did not improve more than tol=%f"
                    " for %d consecutive epochs."
                    % (self.tol, self.n_iter_no_change)
                )
                print(msg)
                break

            if it == self.max_iter:
                msg = (
                    "Stochastic Optimizer: Maximum iterations (%d) "
                    "reached and the optimization hasn't converged yet. "
                    "Loss: %f."
                    % (self.max_iter, self.loss)
                )
                print(msg)

        self.fitted = True
        return self


    def predict(self, X):
        if self.fitted:
            self.activations[0] = X
            self._feed_forward()

            return self.activations[-1]
        else:
            print("Model is not fitted.")
            return 1

    def score(self, X, y):
        return self.r2_score(X, y)

    def r2_score(self, X, y):
        y_pred = self.predict(X)

        u = ((y - y_pred)**2).sum()
        v = ((y - y.mean()) ** 2).sum()

        return 1 - u/v
    
    def mse_score(self, X, y):
        y_pred = self.predict(X)

        return ((y - y_pred)**2).mean()


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_deriv(y):
    return y*(1.-y)

def relu(x):
    return np.maximum(x, 0.)

def relu_deriv(y):
    return np.heaviside(y, 0.)

def leaky_relu(x, leak):
    return np.maximum(leak*x, x)

def leaky_relu_deriv(y, leak):
    return np.heaviside(y, 0.) + leak * np.heaviside(-y, 1.)

def id(x):
    return x

def id_deriv(x):
    return np.zeros(size=x.shape)

def cost_ols(y, pred):
    err = y - pred
    return np.dot(err, err)/len(y)



ACTIVATION = {
    "identity": id,
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu
}

DERIVATIVES = {
    "identity": id_deriv,
    "sigmoid": sigmoid_deriv,
    "relu": relu_deriv,
    "leaky_relu": leaky_relu_deriv
}



class MLPRegNormalInit(MLPRegressor):
    # Adopt MLPRegressor with normally distributed initial weights

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            # loss="squared_error",
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
    
    def _init_coef(self, fan_in, fan_out, dtype):
        # Changed probability distribution to normal (from uniform)
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.normal(
            0., init_bound, (fan_in, fan_out)
        )
        # intercept_init = self._random_state.normal(0, init_bound, fan_out)
        intercept_init = np.zeros(fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init
