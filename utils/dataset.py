import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_files, labels, bboxes, transform):
        self.image_files = image_files
        self.labels = labels
        self.bboxes = bboxes
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        return img, label, bbox

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
    image_id_to_bbox = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_label:
            img_info = next(img for img in data['images'] if img['id'] == img_id)
            width, height = img_info['width'], img_info['height']
            bbox = ann['bbox']
            norm_bbox = [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]
            image_id_to_label[img_id] = ann['category_id']
            image_id_to_bbox[img_id] = norm_bbox

    image_files = [os.path.join(dtrnimg, image_id_to_filename[img_id]) 
                   for img_id in image_id_to_label]
    labels = np.array([image_id_to_label[img_id] for img_id in image_id_to_label])
    bboxes = np.array([image_id_to_bbox[img_id] for img_id in image_id_to_label])
    
    return image_files, labels, bboxes, transform, data