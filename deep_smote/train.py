import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
modpth = os.path.join(base_dir, 'models', 'deep_smote')
os.makedirs(modpth, exist_ok=True)

# Load data
image_files, labels, bboxes, transform, _ = load_data(
    base_dir, 'train', image_size=args['image_size'], n_channel=args['n_channel']
)
print(f"Loaded {len(image_files)} images, label distribution: {collections.Counter(labels)}")
print(f"Images shape: {len(image_files)}, Bboxes shape: {bboxes.shape}")

# DataLoader
dataset = CustomDataset(image_files, labels, bboxes, transform)
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
    tbbox_loss = 0.0
    tdiscr_loss = 0.0

    for images, labs, bboxes_batch in train_loader:
        images, labs, bboxes_batch = images.to(device), labs.to(device), bboxes_batch.to(device).float()
        
        encoder.zero_grad()
        decoder.zero_grad()

        # Forward pass
        z = encoder(images, bboxes_batch)
        x_hat, bbox_hat = decoder(z)

        # Losses
        mse = criterion(x_hat, images)
        bbox_loss = criterion(bbox_hat, bboxes_batch)

        # SMOTE augmentation
        tc = np.random.choice(range(1, 9), 1)[0]
        xclass, yclass, bbox_class = biased_get_class(tc, dec_x, labels, bboxes)
        nsamp = min(len(xclass), 100)
        ind = np.random.choice(len(xclass), nsamp, replace=False)
        xclass, yclass, bbox_class = xclass[ind], yclass[ind], bbox_class[ind]

        xclass = torch.tensor(xclass, dtype=torch.float32).to(device)
        bbox_class = torch.tensor(bbox_class, dtype=torch.float32).to(device)
        with torch.no_grad():
            z_class = encoder(xclass, bbox_class).cpu().numpy()
        
        xsamp, _, bbox_samp = G_SM(z_class[:, :-args['bbox_dim']], yclass, bbox_class.cpu().numpy(), nsamp, tc)
        xsamp = torch.tensor(np.concatenate([xsamp, bbox_samp], axis=1), dtype=torch.float32).to(device)
        ximg, bbox_img = decoder(xsamp)

        mse2 = criterion(ximg, xclass)
        bbox_loss2 = criterion(bbox_img, bbox_class)

        comb_loss = mse + args['bbox_lambda'] * bbox_loss + args['lambda'] * (mse2 + args['bbox_lambda'] * bbox_loss2)
        comb_loss.backward()

        enc_optim.step()
        dec_optim.step()

        train_loss += comb_loss.item() * images.size(0)
        tmse_loss += mse.item() * images.size(0)
        tbbox_loss += bbox_loss.item() * images.size(0)
        tdiscr_loss += (mse2.item() + bbox_loss2.item()) * images.size(0)

    train_loss /= len(train_loader.dataset)
    tmse_loss /= len(train_loader.dataset)
    tbbox_loss /= len(train_loader.dataset)
    tdiscr_loss /= len(train_loader.dataset)
    print(f"Epoch: {epoch} \tTrain Loss: {train_loss:.6f} \tMSE: {tmse_loss:.6f} \tBbox: {tbbox_loss:.6f} \tDiscr: {tdiscr_loss:.6f}")

    # Save losses to CSV
    loss_dict = {
        'train_loss': train_loss,
        'mse_loss': tmse_loss,
        'bbox_loss': tbbox_loss,
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