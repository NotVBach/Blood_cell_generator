import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image
from collections import defaultdict
import json

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

def biased_get_class(dec_x: np.ndarray, dec_y: np.ndarray, c: int):
    """
    Select np.array images by class
    
    Parameters
    ----------
    dec_x : np.ndarray
        Array of images
    dec_y : np.ndarray
        Array of corresponding labels
    c : int
        Class label to filter

    Returns
    -------
    tuple
        Images and labels for the specified class
    """
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X: np.ndarray, y: np.ndarray, n_to_sample: int, cl: int):
    """
    Generate synthetic samples using SMOTE.

    Parameters
    ----------
    X : np.ndarray
        input features (latent space)
    y : np.ndarray
        labels
    n_to_sample : int
        number of samples need to generate
    cl : int
        Class label for generated samples

    Returns
    -------
    tuple
        Synthetic samples and their labels
    """

    # fitting the model
    n_neigh = 5 + 1 # 5 neighbor with the base index itself
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # shape: (n_to_sample, ) with random number from 1 to len(X)
    # Ex: [436 483 417 ... 361 107 212] for len(X)=500
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)

    # shape: (n_to_samples, ) with random number from 1 to n_neigh (6) 
    # Ex: [3 1 1 ... 1 3 2]
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)

    return samples, [cl] * n_to_sample

def compute_imbal(ann_file: str, num_classes: int):
    print("Calculating imbalance difference...")
    if not os.path.exists(ann_file):
        print(f"Error: {ann_file} not found.")
        return [0] * num_classes
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    images_per_category = defaultdict(set)
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_name = category_map.get(category_id, 'unknown')
        images_per_category[category_name].add(image_id)
    
    # Map category names to IDs
    categories = ['ba', 'eo', 'erb', 'ig', 'lym', 'mono', 'neut', 'platelet']
    id_to_category = {i + 1: cat for i, cat in enumerate(categories)}
    category_to_id = {cat: i + 1 for i, cat in enumerate(categories)}
    
    # Count images per category 
    label_counts = {category_to_id[name]: len(images) for name, images in images_per_category.items() if name in category_to_id}
    
    # Initialize imbal list for classes 1 to num_classes
    imbal = [0] * num_classes
    for class_id in range(1, num_classes + 1):
        imbal[class_id - 1] = label_counts.get(class_id, 0)

    # max_count = max(imbal) if imbal else 0
    # # target_count = max(int(max_count))
    # imbal = [max_count for _ in range(num_classes)]
    print(imbal)
    return imbal

def save_images(images: np.ndarray, labels: np.ndarray, output_dir: str, prefix: str = 'synth') -> None:
    """
    Save a batch of images as JPG files, organized by class

    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (num_samples, channels, height, width).
        Assumes images are normalized with mean=0.5, std=0.5.
    labels : np.ndarray
        Array of corresponding class labels (1-based).
    output_dir : str
        Base directory to save images (e.g., 'noaug/synthetic_images/').
    prefix : str, optional
        Prefix for image filenames (default: 'synth').
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = images * 0.5 + 0.5  # Reverse normalization
    images = np.clip(images, 0, 1)  # Ensure values are in [0, 1]
    images = (images * 255).astype(np.uint8)

    for idx, (img, label) in enumerate(zip(images, labels)):
        class_dir = os.path.join(output_dir, f'class_{label}')
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Transpose from (C, H, W) to (H, W, C) for PIL
        img = np.transpose(img, (1, 2, 0))
        
        # Save as JPG
        img_pil = Image.fromarray(img)
        img_path = os.path.join(class_dir, f'{prefix}_{idx}.jpg')
        img_pil.save(img_path, 'JPEG', quality=95)
