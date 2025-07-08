import numpy as np
from scipy.special import comb
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    total = 0
    for i, j in zip(row_ind, col_ind):
        total += w[i, j]
    return total * 1.0 / y_pred.size

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

def ari_score(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def all_metrics(y_true, y_pred):
    return {
        "acc": cluster_acc(y_true, y_pred),
        "nmi": nmi_score(y_true, y_pred),
        "ari": ari_score(y_true, y_pred),
        "ri": rand_index_score(y_pred, y_true)
    }