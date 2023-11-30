class Optimizer(object):
    def __init__(self, lr):
        self.lr = lr

    def update(self, x, dfdx, weight_decay):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        super(SGD, self).__init__(lr)

    def update(self, x, dfdx, weight_decay):
        x -= self.lr * dfdx
        return x


class Momentum(Optimizer):
    def __init__(self, lr, momentum=0.9, v=None):
        super(Momentum, self).__init__(lr)
        self.momentum = momentum
        self.v = v if v is not None else 0

    def update(self, x, dfdx, weight_decay):
        self.v = self.momentum * self.v - self.lr * dfdx
        x += self.v
        return x


class NAG(Optimizer):
    def __init__(self, lr, momentum=0.9, v=None):
        super(NAG, self).__init__(lr)
        self.momentum = momentum
        self.v = v if v is not None else 0

    def update(self, x, dfdx, weight_decay):
        v_prev = self.v
        self.v = self.momentum * self.v - self.lr * dfdx
        x += -self.momentum * v_prev + (1 + self.momentum) * self.v
        return x


class Adagrad(Optimizer):
    def __init__(self, lr, epsilon=1e-7):
        super(Adagrad, self).__init__(lr)
        self.epsilon = epsilon
        self.r = 0

    def update(self, x, dfdx, weight_decay):
        self.r += dfdx ** 2
        adjusted_lr = self.lr / (np.sqrt(self.r) + self.epsilon)
        x -= adjusted_lr * dfdx
        return x
