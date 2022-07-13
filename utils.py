import numpy as np


def regress(X, b0, b1):
    logit = b0 + X @ b1
    return np.clip(logit, -3, 3)


def sigmoid(logit, fast=True):
    if fast:
        return 0.5 + logit / (1 + np.abs(logit)) / 2
    return 1 / (1 + np.exp(-logit))


def soft_threshold(z, r):
    return np.sign(z) * np.clip(abs(z) - r, 0, None)


def cross_entropy(y, p):
    return y @ np.log(p) + (1 - y) @ np.log(1 - p)


def aic(k, L):
    return k - L


def bic(k, L, n):
    return k * np.log(n) - 2 * L


def log_max_lambda(X, y, alpha):
    z = np.where(y == 0, -3, 3)
    maxXy = max(abs(X.T @ z))
    lmbda = maxXy / X.shape[0] / max(1e-1, alpha)
    return np.log(lmbda)
