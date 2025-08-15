import numpy as np
from sklearn.neighbors import NearestNeighbors

def G_SM(X, y, n_to_sample, cl):
    n_neigh = 6
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    alpha = np.random.rand(n_to_sample, 1)
    samples = X_base + alpha * (X_neighbor - X_base)

    return samples, [cl] * n_to_sample

def biased_get_class(c, images, labels):
    mask = labels == c
    return images[mask], labels[mask]