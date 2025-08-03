import numpy as np
from sklearn.neighbors import NearestNeighbors
import collections

def generate_balanced_inputs(labels, bboxes, n_to_sample, cl, n_z, num_classes=8):
    """
    Generate balanced inputs for BAGAN: noise vectors, class labels, and interpolated bboxes.
    
    Args:
        labels: np.array of class labels (1-8).
        bboxes: np.array of normalized bounding boxes [x, y, w, h].
        n_to_sample: Number of samples to generate for class `cl`.
        cl: Target class (1-8).
        n_z: Dimension of noise vector.
        num_classes: Number of classes (default 8).
    
    Returns:
        z: Noise vectors (n_to_sample, n_z).
        class_labels: One-hot encoded class labels (n_to_sample, num_classes).
        bbox_samples: Interpolated bounding boxes (n_to_sample, bbox_dim).
    """
    # Validate input sizes
    if len(labels) != len(bboxes):
        raise ValueError(
            f"Size mismatch: labels ({len(labels)}) and bboxes ({len(bboxes)}) must have the same length"
        )

    # Get samples for the target class
    mask = labels == cl
    if len(mask) != len(bboxes):
        raise ValueError(
            f"Mask size ({len(mask)}) does not match bboxes size ({len(bboxes)})"
        )

    class_bboxes = bboxes[mask]
    
    # If not enough samples, use random bboxes
    if len(class_bboxes) < 2:
        print(f"Warning: Class {cl} has {len(class_bboxes)} samples (< 2), using random bboxes")
        z = np.random.randn(n_to_sample, n_z)
        class_labels = np.zeros((n_to_sample, num_classes))
        class_labels[:, cl-1] = 1
        bbox_samples = np.random.rand(n_to_sample, 4)  # Random normalized bboxes
        bbox_samples = np.clip(bbox_samples, 0, 1)
        return z, class_labels, bbox_samples
    
    # Use NearestNeighbors to interpolate bboxes
    n_neigh = min(6, len(class_bboxes))
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(class_bboxes)
    dist, ind = nn.kneighbors(class_bboxes)

    base_indices = np.random.choice(list(range(len(class_bboxes))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    bbox_base = class_bboxes[base_indices]
    bbox_neighbor = class_bboxes[ind[base_indices, neighbor_indices]]

    alpha = np.random.rand(n_to_sample, 1)
    bbox_samples = bbox_base + alpha * (bbox_neighbor - bbox_base)
    bbox_samples = np.clip(bbox_samples, 0, 1)

    # Generate random noise and class labels
    z = np.random.randn(n_to_sample, n_z)
    class_labels = np.zeros((n_to_sample, num_classes))
    class_labels[:, cl-1] = 1  # One-hot encode class `cl`

    return z, class_labels, bbox_samples