import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import collections
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from balance_gan.models import Generator, Discriminator
from utils.dataset import CustomDataset, load_data
from utils.save_losses import save_losses
from utils.plot_losses import plot_losses
from config import args

# Print CUDA version
print(torch.version.cuda)

t0 = time.time()

# Paths
base_dir = args['base_dir']
modpth = os.path.join(base_dir, 'models', 'balance_gan')
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
generator = Generator(args).to(device)
discriminator = Discriminator(args).to(device)
criterion_bce = nn.BCEWithLogitsLoss().to(device)
criterion_mse = nn.MSELoss().to(device)
criterion_ce = nn.CrossEntropyLoss().to(device)
gen_optim = torch.optim.Adam(generator.parameters(), lr=args['lr'], betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args['disc_lr'], betas=(0.5, 0.999))

# Training loop
best_loss = np.inf
for epoch in range(args['epochs']):
    generator.train()
    discriminator.train()
    g_loss_total = 0.0
    d_loss_total = 0.0
    img_loss_total = 0.0
    bbox_loss_total = 0.0
    class_loss_total = 0.0

    for images, labs, bboxes_batch in train_loader:
        images, labs, bboxes_batch = images.to(device), labs.to(device), bboxes_batch.to(device).float()
        batch_size = images.size(0)

        # Real and fake labels
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # --- Train Discriminator ---
        discriminator.zero_grad()
        
        # Real images
        real_score, class_score, bbox_pred = discriminator(images)
        d_loss_real = criterion_bce(real_score, real_label)
        d_class_loss = criterion_ce(class_score, labs - 1)
        d_bbox_loss = criterion_mse(bbox_pred, bboxes_batch)
        d_loss = d_loss_real + d_class_loss + args['bbox_lambda'] * d_bbox_loss

        # Fake images
        z = torch.randn(batch_size, args['n_z'], device=device)
        fake_images, fake_bboxes = generator(z, labs - 1, bboxes_batch)
        fake_score, fake_class_score, fake_bbox_pred = discriminator(fake_images.detach())
        d_loss_fake = criterion_bce(fake_score, fake_label)
        d_loss += d_loss_fake

        d_loss.backward()
        disc_optim.step()

        # --- Train Generator ---
        generator.zero_grad()
        fake_score, fake_class_score, fake_bbox_pred = discriminator(fake_images)
        g_loss = criterion_bce(fake_score, real_label)
        g_class_loss = criterion_ce(fake_class_score, labs - 1)
        g_bbox_loss = criterion_mse(fake_bboxes, bboxes_batch)
        g_total_loss = g_loss + g_class_loss + args['bbox_lambda'] * g_bbox_loss

        g_total_loss.backward()
        gen_optim.step()

        g_loss_total += g_loss.item() * batch_size
        d_loss_total += d_loss.item() * batch_size
        img_loss_total += g_loss.item() * batch_size
        bbox_loss_total += g_bbox_loss.item() * batch_size
        class_loss_total += g_class_loss.item() * batch_size

    g_loss_total /= len(train_loader.dataset)
    d_loss_total /= len(train_loader.dataset)
    img_loss_total /= len(train_loader.dataset)
    bbox_loss_total /= len(train_loader.dataset)
    class_loss_total /= len(train_loader.dataset)
    print(f"Epoch: {epoch} \tG Loss: {g_loss_total:.6f} \tD Loss: {d_loss_total:.6f} \tImg: {img_loss_total:.6f} \tBbox: {bbox_loss_total:.6f} \tClass: {class_loss_total:.6f}")

    # Save losses to CSV
    loss_dict = {
        'g_loss': g_loss_total,
        'd_loss': d_loss_total,
        'img_loss': img_loss_total,
        'bbox_loss': bbox_loss_total,
        'class_loss': class_loss_total
    }
    save_losses(loss_dict, modpth, 'balance_gan', epoch)

    if g_loss_total < best_loss and args['save']:
        print('Saving..')
        torch.save(generator.state_dict(), os.path.join(modpth, 'bst_gen.pth'))
        torch.save(discriminator.state_dict(), os.path.join(modpth, 'bst_disc.pth'))
        best_loss = g_loss_total

# Plot losses
plot_losses(os.path.join(modpth, 'balance_gan_training_losses.csv'), modpth, 'balance_gan')

t1 = time.time()
print(f"Total time (min): {(t1 - t0) / 60:.2f}")