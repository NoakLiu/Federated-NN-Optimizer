import numpy as np

class BatchNormalization(object):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.global_mean = None
        self.global_var = None
        self.X = None
        self.X_normalized = None
        self.gamma = None
        self.beta = None
        self.dgamma = None
        self.dbeta = None
        self.v_gamma = None
        self.v_beta = None

    def forward(self, X, is_training=True):
        N, D = X.shape

        if self.global_mean is None:
            self.global_mean = np.mean(X, axis=0)
            self.global_var = np.var(X, axis=0)
            self.gamma = np.ones(D, dtype=X.dtype)
            self.beta = np.zeros(D, dtype=X.dtype)
            self.v_gamma = np.zeros(self.gamma.shape)
            self.v_beta = np.zeros(self.beta.shape)

        if is_training:
            sample_mean = np.mean(X, axis=0)
            sample_var = np.var(X, axis=0)

            self.global_mean = self.momentum * self.global_mean + (1 - self.momentum) * sample_mean
            self.global_var = self.momentum * self.global_var + (1 - self.momentum) * sample_var

            X_hat = (X - sample_mean) / np.sqrt(sample_var + self.epsilon)
        else:
            X_hat = (X - self.global_mean) / np.sqrt(self.global_var + self.epsilon)

        out = self.gamma * X_hat + self.beta
        self.X = X
        self.X_normalized = X_hat

        return out

    def backward(self, delta):
        N, D = delta.shape

        x_mu = self.X - self.global_mean
        std_inv = 1. / np.sqrt(self.global_var + self.epsilon)

        dbeta = delta.sum(axis=0)
        dgamma = np.sum(self.X_normalized * delta, axis=0)
        dx_norm = delta * self.gamma
        dvar = np.sum(dx_norm * x_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    def update(self, optimizer, weight_decay):
        self.gamma = optimizer.update(self.gamma, self.dgamma, weight_decay)
        self.beta = optimizer.update(self.beta, self.dbeta, weight_decay)
