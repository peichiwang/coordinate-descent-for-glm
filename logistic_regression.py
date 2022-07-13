import numpy as np
from sklearn.preprocessing import Normalizer
from utils import *


class logisticRegression(object):
    """
    Parameters
    ----------
    alpha : float, default=0.5
        Must be in [0, 1]. Run Lasso regression if alpha is 1 
        and ridge regression when alpha is set to be 0.

    lmbda : list, defualt=None
        Regularization strength. All element must be positive. 
        Default sequence is generated referred to [1].

    iteration : int, default = 10
        Update iteration**2 times under each lmbda.

    References
    ----------
    ..[1] Friedman, Jerome, Trevor Hastie, and Rob Tibshirani. 
    "Regularization paths for generalized linear models via coordinate 
    descent." Journal of statistical software 33.1 (2010): 1.
    """

    def __init__(self, alpha=0.5, lmbda=None, iteration=10):

        assert 0.0 <= alpha <= 1.0
        self.a = alpha
        self.iteration = iteration
        self.status = {i: [] for i in ["b0", "b1", "loss", "aic", "bic"]}
        if lmbda is None:
            self.status["lmbda"] = None
        else:
            self.status["lmbda"] = sorted(lmbda, reverse=True)

    def fit(self, X, y):

        # Initialization
        self.n, self.d = X.shape
        self.y = y
        self.normalizer = Normalizer().fit(X)
        self.X = self.normalizer.transform(X)
        self.X2 = self.X.T @ self.X

        # Default lambda space
        if self.status["lmbda"] is None:
            lmbda = log_max_lambda(self.X, self.y, self.a)
            self.status["lmbda"] = np.logspace(lmbda-3, lmbda + 5, 20)[::-1]

        # run through each lambda
        self.b0 = np.zeros(1)
        self.b1 = np.zeros(self.d)

        for lmbda in self.status["lmbda"]:
            for i in range(self.iteration):
                z = self._working_response(lmbda)
                self.b0 = np.mean(z)
                den = 0.25 * self.n + lmbda * (1 - self.a)
                Xz = self.X.T @ z

                for j in range(self.iteration):
                    self._coordinate_descent(lmbda, den, Xz)

            self.save_status()

        return self

    def _working_response(self, lmbda):
        logit = regress(self.X, self.b0, self.b1)
        prob = sigmoid(logit)
        z = logit + (self.y - prob) * 4
        return z

    def _coordinate_descent(self, lmbda, den, Xz):
        for j in np.random.permutation(self.d):
            num = 0.25 * (Xz[j] - self.X2[j, :] @ self.b1)
            st = soft_threshold(num, lmbda * self.a)
            self.b1[j] = st / den

    def select(self, cri):
        idx = np.argmin(self.status[cri])
        self.b0 = self.status["b0"][idx]
        self.b1 = self.status["b1"][idx]
        return self.status["lmbda"][idx]

    def predict_prob(self, X):
        X = self.normalizer.transform(X)
        logit = regress(X, self.b0, self.b1)
        return sigmoid(logit, fast=False)

    def predict(self, X):
        prob = self.predict_prob(X)
        return (prob > 0.5).astype(int)

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

    def save_status(self):
        k = np.sum(self.b1 != 0)
        logit = regress(self.X, self.b0, self.b1)
        prob = sigmoid(logit, fast=False)
        log_like = cross_entropy(self.y, prob)

        self.status["b0"].append(self.b0.copy())
        self.status["b1"].append(self.b1.copy())
        self.status["loss"].append(-log_like)
        self.status["aic"].append(aic(k, log_like))
        self.status["bic"].append(bic(k, log_like, self.n))
