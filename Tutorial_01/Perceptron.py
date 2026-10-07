import numpy as np


class Perceptron(object):
    """Simple perceptron built from scratch (Tutorial 1, Steps 1-4)."""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta          # learning rate
        self.n_iter = n_iter    # number of passes over the training data

    def weighted_sum(self, X):
        # np.dot(X, w[1:]) is the dot product of inputs and weights,
        # w[0] is the bias term added to it
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # Step function: 1 if weighted sum >= 0, otherwise -1
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        # initialize weights to 0 (one extra element for the bias)
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []  # number of misclassifications in each iteration

        print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                # 1. predicted value using current weights
                y_pred = self.predict(xi)

                # 2. weight update: eta * (y - y_pred)
                update = self.eta * (target - y_pred)

                # 3. update feature weights: Wi = Wi + update * Xi
                self.w_[1:] = self.w_[1:] + update * xi
                print("Updated Weights:", self.w_[1:])

                # update the bias (X0 = 1)
                self.w_[0] = self.w_[0] + update

                # count a mistake if an update happened
                error += int(update != 0.0)

            self.errors_.append(error)
        return self


class PerceptronBinary(Perceptron):
    """Task 3: same perceptron, but labels are 1 and 0 instead of 1 and -1."""

    def predict(self, X):
        # Step function now outputs 1 or 0
        return np.where(self.weighted_sum(X) >= 0.0, 1, 0)
