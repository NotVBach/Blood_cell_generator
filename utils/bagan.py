import numpy as np
from collections import Counter

def generate_balanced_inputs(labels, n_to_sample, cl, n_z, num_classes):
    z = np.random.randn(n_to_sample, n_z, 1, 1)
    class_labels = np.full(n_to_sample, cl, dtype=np.int64)
    return z, class_labels