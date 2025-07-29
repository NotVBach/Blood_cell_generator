import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import torch
import torch.nn as nn

def free_params(module: nn.Module):
    """
    Unfreeze all parameters of a module.
    """
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    """
    Freeze all parameters of a module.
    """
    for p in module.parameters():
        p.requires_grad = False

def G_SM(X, y, bboxes, n_to_sample, cl):
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # shape: (n_to_sample, ) with random number from 1 to len(X)
    # Ex: [436 483 417 ... 361 107 212] for len(X)=500
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    # shape: (n_to_samples, ) with random number from 1 to n_neigh (6) 
    # Ex: [3 1 1 ... 1 3 2]
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)