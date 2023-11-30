import numpy as np


class Activation(object):
    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_deriv(self, a):
        return 1 * (a >= 0)

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        return a * (1 - a)

    def __sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def __sigmoid_deriv(self, a):
        return np.exp(-a) / ((1 + np.exp(-a)) ** 2)

    def __softmax(self, x):
        mx = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - mx)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        return x_exp / x_sum

    def __softmax_deriv(self, y, y_pred):
        return y_pred - y

    def __init__(self, activation='relu'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv
        elif activation == 'sigmoid':
            self.f = self.__sigmoid
            self.f_deriv = self.__sigmoid_deriv

    def forward(self, x):
        x_out = self.f(x)
        return x_out

    def backward(self, delta):
        delta = self.f_deriv(delta) * delta
        return delta
