from math import log, sqrt
from random import choices
import numpy as np

from sklearn.linear_model import LinearRegression


def PolyWeights(experts, observe, loss, T, lr=None):
    N = len(experts)
    lr = log(N) / T if not lr else lr

    weights = np.ones((T, N))
    total_loss = 0

    for t in range(T):
        selected = choices(experts, weights=weights)
        outcome = observe(t)
        f = lambda wt: wt * (1 - lr * (loss(wt, outcome)))
        weights = np.apply_along_axis(f, t, weights)
        total_loss += loss(selected, outcome)

    # Also return empirical average of weights or alternatively, a N x T matrix to sample uniformly from
    return weights, total_loss


def LinReg(X, y, wts):
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=wts)
    return reg


def filter_grp(x, y, g_):
    return x[g_], y[g_]


def eps_k(h, L, x_k, y_k):
    return L(y_k, h.predict(x_k))


def weights_updater(lr, err, old_wts, g_):
    old_wts[g_] = old_wts[g_] * np.exp(lr * err)
    return old_wts


def MinimaxFair(X, y, G, L, T, H_, lr=None):
    H = [None for _ in range(T)]
    N = len(y)
    K = len(G)
    lr = lr if lr else np.log(N) / T
    weights = np.ones(N) / N

    for t in range(T):
        h_t = H_(X, y, weights)
        H[t] = h_t
        errs = [None for _ in range(K)]
        for k in range(K):
            err = eps_k(h_t, L, *filter_grp(X, y, G[k]))
            weights = weights_updater(lr, err, weights, G[k])
            errs[k] = err
    return H
