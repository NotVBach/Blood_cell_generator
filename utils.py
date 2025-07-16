# utils.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def biased_get_class(dec_x, dec_y, c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X, y, n_to_sample, cl):
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)
    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)
    return samples, [cl] * n_to_sample

def biased_get_class1(c):
    # Note: This assumes dec_x and dec_y are globally available or passed differently
    # In generate_samples.py, these are passed as arguments, so this is a placeholder
    # We'll rely on generate_samples.py passing dec_x and dec_y explicitly
    def inner(dec_x, dec_y):
        xbeg = dec_x[dec_y == c]
        ybeg = dec_y[dec_y == c]
        return xbeg, ybeg
    return inner

# def G_SM1(X, y, n_to_sample, cl):
#     n_neigh = 5 + 1
#     nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
#     nn.fit(X)
#     dist, ind = nn.kneighbors(X)
#     base_indices = np.random.choice(list(range(len(X))), n_to_sample)
#     neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)
#     X_base = X[base_indices]
#     X_neighbor = X[ind[base_indices, neighbor_indices]]
#     samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)
#     return samples, [cl] * n_to_sample