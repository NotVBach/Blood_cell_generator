import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import args
import collections

class DatasetLoader(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = {img['id']: img['file_name'] for img in self.annotations['images']}
        self.labels = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
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
        
        label = self.labels.get(img_id, 0)
        return image, label

def load_data(data_dir: str, split='train'):
    """
    Load dataset path from json files.

    Parameters
    ----------
    data_dir : str
        dataset's directory path
    split : str
        in case there there are other set in future (test, val, train1, train2, ...)

    Returns
    -------
    str
        Images path and annotation path
    """
    img_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(data_dir, 'annotations', f'{split}.json')
    return img_dir, ann_file

def preprocess_data(img_dir: str, ann_file: str):
    """
    Resize images then generate image and label numpy array

    Parameters
    ----------
    img_dir : str
        images directory path
    ann_file : str
        annotation file (json) path

    Returns
    -------
    np.array
        Images and annotation array
    """
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = DatasetLoader(img_dir, ann_file, transform=transform)
    images, labels = [], []
    
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f'images shape: {images.shape}')
    print(f' labels shape: {labels.shape}')
    print(f'Label distribution: {collections.Counter(labels)}')
    
    return images, labels

def create_dataloader(img_dir, ann_file, batch_size, num_workers: int = 0):
    """
    Resize images then generate dataset using torch DataLoader

    Parameters
    ----------
    img_dir : str
        images directory path
    ann_file : str
        annotation file (json) path
    batch_size: int
        batch_size (can be alter in config)
    num_workers: int
        load data in parallel using multiple subprocesses but not gonna use it anyway

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        torch like dataset
    """
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = DatasetLoader(img_dir, ann_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Verify batch shapes
    # for images, labels in dataloader:
    #     print(f'Batch images shape: {images.shape}, labels shape: {labels.shape}')
    #     break  
    return dataloader