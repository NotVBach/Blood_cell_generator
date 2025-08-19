# utils/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import numpy as np
from config import args

class CustomDataset(Dataset):
    def __init__(self, image_files, labels, transform):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_data(base_dir, split='train', image_size=(32, 32), n_channel=3):
    dtrnimg = os.path.join(base_dir, split)
    dtrnlab = os.path.join(base_dir, 'annotations', f'{split}.json')
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channel, (0.5,) * n_channel)
    ])

    with open(dtrnlab, 'r') as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    image_id_to_label = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        label = ann['category_id']
        if img_id in image_id_to_filename and 1 <= label <= args['num_classes']:  # Validate label
            image_id_to_label[img_id] = label

    valid_image_ids = list(image_id_to_label.keys())
    image_files = [os.path.join(dtrnimg, image_id_to_filename[img_id]) 
                   for img_id in valid_image_ids]
    labels = np.array([image_id_to_label[img_id] for img_id in valid_image_ids])

    assert len(image_files) == len(labels), (
        f"Array size mismatch: image_files ({len(image_files)}), labels ({len(labels)})"
    )
    invalid_labels = [l for l in labels if l < 1 or l > 8]
    if invalid_labels:
        print(f"Warning: Found invalid labels: {invalid_labels}")
        raise ValueError("Labels must be in range [1, 8]")

    print(f"Loaded {len(image_files)} images with annotations")
    return image_files, labels, transform, data