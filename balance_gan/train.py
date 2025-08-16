import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from balance_gan.models import Autoencoder, Generator, Discriminator
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
autoencoder = Autoencoder(args).to(device)
generator = Generator(args).to(device)
discriminator = Discriminator(args).to(device)

# Autoencoder optimizer
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args['lr'], betas=(0.5, 0.999))
criterion_mse = nn.MSELoss().to(device)

# GAN optimizers
criterion_bce = nn.BCEWithLogitsLoss().to(device)
criterion_ce = nn.CrossEntropyLoss().to(device)
gen_optim = torch.optim.Adam(generator.parameters(), lr=args['lr'], betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args['disc_lr'], betas=(0.5, 0.999))

# Pre-train autoencoder
print("Pre-training autoencoder...")
autoencoder.train()
for epoch in range(10):  # 10 epochs for pre-training
    ae_loss_total = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        ae_optimizer.zero_grad()
        x_recon = autoencoder(images)
        ae_loss = criterion_mse(x_recon, images)
        ae_loss.backward()
        ae_optimizer.step()
        ae_loss_total += ae_loss.item() * images.size(0)
    ae_loss_total /= len(train_loader.dataset)
    print(f"AE Epoch: {epoch} \tAE Loss: {ae_loss_total:.6f}")

# Initialize generator with autoencoder's decoder weights
generator_dict = generator.state_dict()
ae_dict = autoencoder.state_dict()
# Copy deconv layers (shared structure)
for key in generator_dict.keys():
    if 'deconv' in key:
        generator_dict[key] = ae_dict[f'decoder.{key.split("deconv.")[-1]}']
generator.load_state_dict(generator_dict)

# Training loop for GAN
best_loss = np.inf
for epoch in range(args['epochs']):
    generator.train()
    discriminator.train()
    g_loss_total = 0.0
    d_loss_total = 0.0
    img_loss_total = 0.0
    class_loss_total = 0.0

    for images, labs in train_loader:
        images, labs = images.to(device), labs.to(device)
        batch_size = images.size(0)

        # Real and fake labels
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # --- Train Discriminator ---
        discriminator.zero_grad()
        
        # Real images
        real_score, class_score = discriminator(images)
        d_loss_real = criterion_bce(real_score, real_label)
        d_class_loss = criterion_ce(class_score, labs - 1)
        d_loss = d_loss_real + d_class_loss

        # Fake images
        z = torch.randn(batch_size, args['n_z'], device=device)
        fake_images = generator(z, labs - 1)
        fake_score, fake_class_score = discriminator(fake_images.detach())
        d_loss_fake = criterion_bce(fake_score, fake_label)
        d_loss += d_loss_fake

        d_loss.backward()
        disc_optim.step()

        # --- Train Generator ---
        generator.zero_grad()
        fake_score, fake_class_score = discriminator(fake_images)
        g_loss = criterion_bce(fake_score, real_label)
        g_class_loss = criterion_ce(fake_class_score, labs - 1)
        g_total_loss = g_loss + g_class_loss

        g_total_loss.backward()
        gen_optim.step()

        g_loss_total += g_loss.item() * batch_size
        d_loss_total += d_loss.item() * batch_size
        img_loss_total += g_loss.item() * batch_size
        class_loss_total += g_class_loss.item() * batch_size

    g_loss_total /= len(train_loader.dataset)
    d_loss_total /= len(train_loader.dataset)
    img_loss_total /= len(train_loader.dataset)
    class_loss_total /= len(train_loader.dataset)
    print(f"Epoch: {epoch} \tG Loss: {g_loss_total:.6f} \tD Loss: {d_loss_total:.6f} \tImg: {img_loss_total:.6f} \tClass: {class_loss_total:.6f}")

    # Save losses to CSV
    loss_dict = {
        'g_loss': g_loss_total,
        'd_loss': d_loss_total,
        'img_loss': img_loss_total,
        'class_loss': class_loss_total
    }
    save_losses(loss_dict, modpth, 'balance_gan', epoch)

    if g_loss_total < best_loss and args['save']:
        print('Saving..')
        torch.save(autoencoder.state_dict(), os.path.join(modpth, 'bst_ae.pth'))
        torch.save(generator.state_dict(), os.path.join(modpth, 'bst_gen.pth'))
        torch.save(discriminator.state_dict(), os.path.join(modpth, 'bst_disc.pth'))
        best_loss = g_loss_total

# Plot losses
plot_losses(os.path.join(modpth, 'balance_gan_training_losses.csv'), modpth, 'balance_gan')

t1 = time.time()
print(f"Total time (min): {(t1 - t0) / 60:.2f}")