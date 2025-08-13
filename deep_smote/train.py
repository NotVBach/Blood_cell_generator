import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from deep_smote.models import Encoder, Decoder
from utils.smote import G_SM, biased_get_class
from utils.dataset import CustomDataset, load_data
from utils.save_losses import save_losses
from utils.plot_losses import plot_losses
from config import args

# Print CUDA version
print(torch.version.cuda)

t0 = time.time()

# Paths
base_dir = args['base_dir']
modpth = os.path.join(args['output_dir'], 'models')
os.makedirs(modpth, exist_ok=True)

# Load data
image_files, labels, transform, _ = load_data(
    base_dir, 'train', image_size=args['image_size'], n_channel=args['n_channel']
)
print(f"Loaded {len(image_files)} images, label distribution: {collections.Counter(labels)}")
print(f"Images shape: {len(image_files)}, Labels shape: {len(labels)}")

# DataLoader
dataset = CustomDataset(image_files, labels, transform)
train_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

# Models and optimizers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
encoder = Encoder(args).to(device)
decoder = Decoder(args).to(device)
criterion = nn.MSELoss().to(device)
enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

# Load images for SMOTE
dec_x = np.array([transform(Image.open(p).convert('RGB')).numpy() for p in image_files])

# Training loop
best_loss = np.inf
for epoch in range(args['epochs']):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    tmse_loss = 0.0
    tdiscr_loss = 0.0

    for images, labs in train_loader:
        images, labs = images.to(device), labs.to(device)
        
        encoder.zero_grad()
        decoder.zero_grad()

        # Forward pass
        z = encoder(images)
        x_hat = decoder(z)

        # Losses
        mse = criterion(x_hat, images)

        # SMOTE augmentation
        tc = np.random.choice(range(1, 9), 1)[0]
        xclass, yclass = biased_get_class(tc, dec_x, labels)
        nsamp = min(len(xclass), 100)
        ind = np.random.choice(len(xclass), nsamp, replace=False)
        xclass, yclass = xclass[ind], yclass[ind]

        xclass = torch.tensor(xclass, dtype=torch.float32).to(device)
        with torch.no_grad():
            z_class = encoder(xclass).cpu().numpy()
        
        xsamp, _ = G_SM(z_class, yclass, nsamp, tc)
        xsamp = torch.tensor(xsamp, dtype=torch.float32).to(device)
        ximg = decoder(xsamp)

        mse2 = criterion(ximg, xclass)

        comb_loss = mse + args['lambda'] * mse2
        comb_loss.backward()

        enc_optim.step()
        dec_optim.step()

        train_loss += comb_loss.item() * images.size(0)
        tmse_loss += mse.item() * images.size(0)
        tdiscr_loss += mse2.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    tmse_loss /= len(train_loader.dataset)
    tdiscr_loss /= len(train_loader.dataset)
    print(f"Epoch: {epoch} \tTrain Loss: {train_loss:.6f} \tMSE: {tmse_loss:.6f} \tDiscr: {tdiscr_loss:.6f}")

    # Save losses to CSV
    loss_dict = {
        'train_loss': train_loss,
        'mse_loss': tmse_loss,
        'discr_loss': tdiscr_loss
    }
    save_losses(loss_dict, modpth, 'deep_smote', epoch)

    if train_loss < best_loss and args['save']:
        print('Saving..')
        torch.save(encoder.state_dict(), os.path.join(modpth, 'bst_enc.pth'))
        torch.save(decoder.state_dict(), os.path.join(modpth, 'bst_dec.pth'))
        best_loss = train_loss

# Plot losses
plot_losses(os.path.join(modpth, 'deep_smote_training_losses.csv'), modpth, 'deep_smote')

t1 = time.time()
print(f"Total time (min): {(t1 - t0) / 60:.2f}")