class Dropout(object):
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = np.random.binomial(n=1, p=1 - self.dropout_prob, size=x.shape)
            result = x * self.mask
            return result / (1 - self.dropout_prob)
        else:
            return x

    def backward(self, delta):
        delta = delta * self.mask / (1 - self.dropout_prob)
        return delta
