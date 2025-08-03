import numpy as np
from sklearn.neighbors import NearestNeighbors

def G_SM(X, y, bboxes, n_to_sample, cl):
    n_neigh = 6
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    bbox_base = bboxes[base_indices]
    bbox_neighbor = bboxes[ind[base_indices, neighbor_indices]]

    alpha = np.random.rand(n_to_sample, 1)
    samples = X_base + alpha * (X_neighbor - X_base)
    bbox_samples = bbox_base + alpha * (bbox_neighbor - bbox_base)

    bbox_samples = np.clip(bbox_samples, 0, 1)
    return samples, [cl] * n_to_sample, bbox_samples

def biased_get_class(c, images, labels, bboxes):
    mask = labels == c
    return images[mask], labels[mask], bboxes[mask]