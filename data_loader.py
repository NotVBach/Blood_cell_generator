import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import collections

def load_mnist_data(img_path, lab_path, batch_size=100, num_workers=0):
    # List image and label files
    ids = os.listdir(img_path)
    idtri_f = [os.path.join(img_path, image_id) for image_id in ids]
    ids = os.listdir(lab_path)
    idtrl_f = [os.path.join(lab_path, image_id) for image_id in ids]

    return idtri_f, idtrl_f

def preprocess_data(img_file, lab_file):
    # Load and preprocess data
    dec_x = np.loadtxt(img_file)
    dec_y = np.loadtxt(lab_file)
    
    print('train imgs before reshape ', dec_x.shape)
    print('train labels ', dec_y.shape)
    print(collections.Counter(dec_y))
    
    # Reshape images to (N, C, H, W)
    dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)
    print('train imgs after reshape ', dec_x.shape)
    
    return dec_x, dec_y

def create_dataloader(dec_x, dec_y, batch_size=100, num_workers=0):
    # Convert to PyTorch tensors
    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader