import numpy as np
import collections

def generate_balanced_inputs(labels, n_to_sample, cl, n_z, num_classes):
    z = np.random.randn(n_to_sample, n_z)
    class_labels = np.zeros((n_to_sample, num_classes))
    class_labels[:, cl-1] = 1
    return z, class_labels
