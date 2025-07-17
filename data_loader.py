import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import args
import collections

class LoadDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = {img['id']: img['file_name'] for img in self.annotations['images']}
        self.labels = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            # Use the first category_id per image for classification
            if img_id not in self.labels:
                self.labels[img_id] = ann['category_id']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_path = os.path.join(self.img_dir, self.images[img_id])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels.get(img_id, 0)  # Default to 0 if no label
        return image, label

def load_noaug_data(data_dir, split):
    img_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(data_dir, 'annotations', f'{split}.json')
    return img_dir, ann_file

def preprocess_data(img_dir, ann_file):
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = LoadDataset(img_dir, ann_file, transform=transform)
    images, labels = [], []
    
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl)
    
    images = np.array(images)
    labels = np.array(labels)
    
    split = 'Train'
    print(f'{split} images shape: {images.shape}')
    print(f'{split} labels shape: {labels.shape}')
    print(f'Label distribution: {collections.Counter(labels)}')
    
    return images, labels

def create_dataloader(img_dir, ann_file, batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),  # Converts to [0, 1], float32
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scales to [-1, 1]
    ])
    
    dataset = LoadDataset(img_dir, ann_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Load data successfully... \n")
    return dataloader