import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
from collections import defaultdict
import os
from PIL import Image

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

def compute_imbal(ann_file, num_classes):
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

    max_count = max(imbal) if imbal else 0
    # target_count = max(int(max_count))
    imbal = [max_count for _ in range(num_classes)]
    
    print(f'Label distribution: {label_counts}')
    print(f'Computed imbal: {imbal}')
    return imbal

def save_images(images, labels, output_dir, image_size, prefix='class'):
    os.makedirs(output_dir, exist_ok=True)
    
    images = (images * 0.5 + 0.5) * 255
    images = images.astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        filename = f'{prefix}_{int(label)}_{i:04d}.jpg'
        filepath = os.path.join(output_dir, filename)
        img_pil = Image.fromarray(img)
        img_pil.save(filepath, 'JPEG')