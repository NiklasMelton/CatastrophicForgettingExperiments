import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state  # Utility to handle random state consistently

class CustomMLPClassifier(MLPClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
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
        # Pass all parameters to the base class
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
        self._weights_initialized = False  # Track if weights have been initialized
        self._random_state = check_random_state(random_state)  # Store the random state

    def _initialize_weights_xavier(self):
        """Custom Xavier initialization for weights and biases."""
        for layer_idx in range(len(self.coefs_)):
            n_in = self.coefs_[layer_idx].shape[0]
            n_out = self.coefs_[layer_idx].shape[1]
            limit = np.sqrt(6 / (n_in + n_out))

            # Xavier initialization for weights using the fixed random state
            self.coefs_[layer_idx] = self._random_state.uniform(
                -limit, limit, size=self.coefs_[layer_idx].shape
            )

            # Initialize biases to small values using the fixed random state
            self.intercepts_[layer_idx] = self._random_state.uniform(
                -0.01, 0.01, size=self.intercepts_[layer_idx].shape
            )

    def partial_fit(self, X, y, classes=None):
        # Initialize weights only on the first call to partial_fit
        if not self._weights_initialized:
            # Call the parent class partial_fit to initialize the internal structure
            super().partial_fit(X, y, classes=classes)
            # Apply custom Xavier initialization
            self._initialize_weights_xavier()
            self._weights_initialized = True  # Mark as initialized
        return super().partial_fit(X, y, classes=classes)
