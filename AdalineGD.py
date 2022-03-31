from Perceptron import Perceptron
import numpy as np


class AdalineGD(Perceptron):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        super().__init__(eta, n_iter, random_state)
        self.cost_ = None

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    @staticmethod
    def activation(X):
        return X
