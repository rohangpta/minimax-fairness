from math import log, sqrt
from random import choices
import numpy as np

from sklearn.linear_model import LinearRegression, SGDClassifier
from random import choice

mapping, columns = None, None


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


def generate_comp_table(x, columns):
    grp_list = columns
    x = x.filter(items=grp_list)
    mapping = {}
    for item in x.columns:
        mapping[item] = x[item].value_counts() / len(x[item])
    return mapping


def G_(model, x, y, n):
    preds = np.array(model.predict(x).round(), dtype="int64")
    y = np.array(y, dtype="int64")
    wrong = y ^ preds
    wrong_idx = np.nonzero(wrong)

    dataset = x.loc[wrong_idx]
    filt = dataset.filter(items=columns)
    size = len(filt)
    op = []
    for col in filt.columns:
        dist = filt[col].value_counts() / size
        rdist = mapping[col][dist.index]
        rdist = (rdist) / rdist.sum()
        diff = dist - rdist

        if not len(diff):
            continue
        argmax = diff.idxmax()

        if 500 <= dist.loc[argmax] * size <= (0.6 * size):
            op.append([col, argmax, max(diff)])
    top_n = sorted(op, key=lambda x: x[2], reverse=True)[:n]
    return [np.where(x[col] == argmax) for col, argmax, _ in top_n]


def LinReg(X, y, wts):
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=wts)
    return reg


def SGD(X, y, wts):
    reg = SGDClassifier()
    reg.fit(X, y, sample_weight=wts)
    return reg


def filter_grp(x, y, g_):
    return x[g_], y[g_]


def eps_k(h, L, x_k, y_k, round=False):
    return L(y_k, h.predict(x_k) if not round else h.predict(x_k).round())


def eps_k_rand(H, L, x_k, y_k, round=False):
    N = len(y_k)
    models = [choice(H) for _ in range(N)]
    preds = [
        (
            models[i].predict(x_k[i].reshape(1, -1))
            if not round
            else models[i].predict(x_k[i].reshape(1, -1)).round()
        ).item()
        for i in range(N)
    ]
    return L(y_k, preds)


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


def test(H, X, y, G, L, eps):
    K = len(G)
    errs = [None for _ in range(K)]
    for k in range(K):
        err = eps(H, L, *filter_grp(X, y, G[k]))
        errs[k] = err
    return errs
