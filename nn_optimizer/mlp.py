import numpy as np
import math
import time
from .layers import LinearLayer
from .activation_functions import Activation
from .dropout import Dropout
from .batch_normalization import BatchNormalization
from .optimizers import SGD


class MLP:
    def __init__(self, n_in, n_out, layers, optimizer=None, activation='relu', activation_last_layer='softmax',
                 dropout_ratio=0, is_batch_normalization=False):
        self.layers = []
        self.activation = activation
        self.optimizer = optimizer
        self.n_in = n_in
        self.n_out = n_out

        for i in range(len(layers)):
            if i == 0:
                self.layers.append(LinearLayer(n_in, layers[i]))
            else:
                self.layers.append(LinearLayer(layers[i - 1], layers[i]))

            if dropout_ratio != 0:
                self.layers.append(Dropout(dropout_ratio))

            if is_batch_normalization:
                self.layers.append(BatchNormalization())

            if i != 0:
                self.layers.append(Activation(activation))

        self.layers.append(LinearLayer(layers[-1], n_out))
        self.layers.append(Activation(activation_last_layer))

    def forward(self, input_v):
        for layer in self.layers:
            output = layer.forward(input_v)
            input_v = output
        return output

    def backward(self, delta):
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def criterion_cross_entropy(self, y, y_hat):
        y_actual_onehot = np.eye(self.n_out)[y].reshape(-1, self.n_out)
        y_predicted = np.clip(y_hat, 1e-12, 1 - 1e-12)
        loss = -np.sum(np.multiply(y_actual_onehot, np.log(y_predicted)), axis=1)
        delta = self.layers[-1].f_deriv(y_actual_onehot, y_predicted)
        return loss, delta

    def update(self, lr, weight_decay):
        optimizer_object = SGD(lr)
        for layer in self.layers:
            if layer.__class__.__name__ == 'LinearLayer' or layer.__class__.__name__ == 'BatchNormalization':
                layer.update(optimizer_object, weight_decay)

    def fit(self, X, y, is_shuffle=False, learning_rate=0.001, epochs=100, batch_size=100, weight_decay=0.95):
        X = np.array(X)
        y = np.array(y)
        to_return = np.zeros(epochs)
        n_batch = math.ceil(X.shape[0] / batch_size)
        batch_loss = np.zeros(n_batch)

        for k in range(epochs):
            if is_shuffle:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]

            for j in range(n_batch):
                X_batch = X[j * batch_size:(j + 1) * batch_size] if j != n_batch - 1 else X[j * batch_size:]
                y_batch = y[j * batch_size:(j + 1) * batch_size] if j != n_batch - 1 else y[j * batch_size:]

                y_batch_hat = self.forward(X_batch)
                loss, delta = self.criterion_cross_entropy(y_batch, y_batch_hat)
                self.backward(delta)
                self.update(learning_rate, weight_decay)

                batch_loss[j] = np.sum(loss)

            to_return[k] = np.mean(batch_loss)
            accuracy = np.sum(np.argmax(self.forward(X), axis=1) == y) / y.shape[0]
            print("Epoch {} loss {:.6f}, accuracy {:.6f}% ".format(k, to_return[k], 100 * accuracy))
        return to_return

    def predict(self, x):
        x = np.array(x)
        return np.argmax(self.forward(x), axis=1)
