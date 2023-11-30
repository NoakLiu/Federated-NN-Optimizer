import numpy as np


class LinearLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None):
        self.input_v = None
        self.output = None
        np.random.seed(2022)
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        self.b = np.zeros(n_out, )
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)

    def forward(self, input_v):
        self.input_v = input_v
        self.output = np.dot(input_v, self.W) + self.b
        return self.output

    def backward(self, delta):
        self.grad_W = np.atleast_2d(self.input_v).T.dot(np.atleast_2d(delta))
        self.grad_b = np.mean(delta, axis=0)
        delta = np.dot(delta, self.W.T)
        return delta

    def update(self, optimizer, weight_decay):
        self.W = optimizer.update(self.W, self.grad_W, weight_decay)
        self.b = optimizer.update(self.b, self.grad_b, weight_decay)
