# strategies.py
import numpy as np
from sklearn.decomposition import NMF


def random_gaussian(d_model, n_heads, rank):
    W = np.random.randn(n_heads * rank, d_model)
    return W

def init_random_orthogonal(d_model, n_heads, rank):
    W = np.random.randn(n_heads * rank, d_model)
    Q, _ = np.linalg.qr(W.T)
    return Q[:, :n_heads * rank].T

def init_svd_sqrt(W_K_head, rank):
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]

def init_svd_full(W_K_head, rank):
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return np.diag(S[:rank]) @ Vt[:rank, :]

def init_svd_noscale(W_K_head, rank):
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return Vt[:rank, :]

def init_pca(W_K_head, rank):
    centered = W_K_head - W_K_head.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]

def init_nmf(W_K_head, rank):
    positive_matrix = np.maximum(W_K_head, 0.0)
    nmf = NMF(n_components=rank, init="nndsvda", random_state=0, max_iter=1000)
    nmf.fit(positive_matrix)
    return nmf.components_

