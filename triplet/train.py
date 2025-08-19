import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from triplet.models import Encoder, Decoder, Discriminator
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
discriminator = Discriminator(args).to(device)
criterion_mse = nn.MSELoss().to(device)
criterion_bce = nn.BCEWithLogitsLoss().to(device)
criterion_triplet = nn.TripletMarginLoss(margin=args['triplet_margin']).to(device)
enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'], betas=(0.5, 0.999))
dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'], betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args['disc_lr'], betas=(0.5, 0.999))

# Training loop
best_loss = np.inf
for epoch in range(args['epochs']):
    encoder.train()
    decoder.train()
    discriminator.train()
    enc_loss_total = 0.0
    dec_loss_total = 0.0
    disc_loss_total = 0.0
    mse_loss_total = 0.0
    triplet_loss_total = 0.0
    batch_count = 0

    for images, labs in train_loader:
        images, labs = images.to(device), labs.to(device)
        labs = labs - 1 
        # if not (labs >= 0).all() or not (labs < 8).all():
        #     print(f"Invalid labels in batch: {labs}")
        #     continue
        batch_size = images.size(0)
        # print(f"Batch size: {batch_size}, Images shape: {images.shape}, Labels shape: {labs.shape}")

        # --- Select anchor, positive, negative samples for triplet loss ---
        anchor_idx = []
        pos_idx = []
        neg_idx = []
        for i in range(batch_size):
            pos_candidates = (labs == labs[i]).nonzero(as_tuple=True)[0]
            neg_candidates = (labs != labs[i]).nonzero(as_tuple=True)[0]
            if len(pos_candidates) > 1 and len(neg_candidates) > 0:
                pos_idx_i = pos_candidates[pos_candidates != i][torch.randint(0, len(pos_candidates)-1, (1,))].item()
                neg_idx_i = neg_candidates[torch.randint(0, len(neg_candidates), (1,))].item()
                anchor_idx.append(i)
                pos_idx.append(pos_idx_i)
                neg_idx.append(neg_idx_i)

        # if not anchor_idx:  # Skip if no valid triplets
        #     print(f"Skipping batch: No valid triplets (labels: {labs.cpu().numpy()})")
        #     continue

        anchor_idx = torch.tensor(anchor_idx, device=device)
        pos_idx = torch.tensor(pos_idx, device=device)
        neg_idx = torch.tensor(neg_idx, device=device)
        # print(f"Triplet sizes: Anchor {len(anchor_idx)}, Positive {len(pos_idx)}, Negative {len(neg_idx)}")

        # --- Train Encoder with Triplet Loss + MSE ---
        # Enable encoder gradients
        for param in encoder.parameters():
            param.requires_grad = True
        encoder.train()

        encoder.zero_grad()
        decoder.zero_grad()

        anchor_z = encoder(images[anchor_idx])
        pos_z = encoder(images[pos_idx])
        neg_z = encoder(images[neg_idx])
        # print(f"Latent shapes: Anchor {anchor_z.shape}, Positive {pos_z.shape}, Negative {neg_z.shape}")
        triplet_loss = criterion_triplet(anchor_z, pos_z, neg_z)

        # Reconstruction loss
        decoded_images = decoder(anchor_z, labs[anchor_idx])
        # print(f"Decoded images shape: {decoded_images.shape}")
        mse_loss = criterion_mse(decoded_images, images[anchor_idx])

        enc_loss = triplet_loss + args['lambda'] * mse_loss
        enc_loss.backward()
        enc_optim.step()
        dec_optim.step()

        # --- Freeze Encoder for Decoder and Discriminator Training ---
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

        # --- Train Decoder with MSE + Discriminator Loss ---
        decoder.zero_grad()
        z = encoder(images)
        # print(f"Encoder output z shape: {z.shape}")
        fake_images = decoder(z, labs)
        # print(f"Fake images shape: {fake_images.shape}")
        fake_score = discriminator(fake_images)
        # print(f"Fake score shape: {fake_score.shape}")
        real_label = torch.ones(batch_size, 1, device=device)
        adv_loss = criterion_bce(fake_score, real_label)
        mse_loss_dec = criterion_mse(fake_images, images)
        dec_loss = mse_loss_dec + args['lambda'] * adv_loss
        dec_loss.backward()
        dec_optim.step()

        # --- Train Discriminator ---
        discriminator.zero_grad()
        real_score = discriminator(images)
        # print(f"Real score shape: {real_score.shape}")
        fake_score = discriminator(fake_images.detach())
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)
        real_loss = criterion_bce(real_score, real_label)
        fake_loss = criterion_bce(fake_score, fake_label)
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        disc_optim.step()

        enc_loss_total += enc_loss.item() * len(anchor_idx)
        dec_loss_total += dec_loss.item() * batch_size
        disc_loss_total += disc_loss.item() * batch_size
        mse_loss_total += mse_loss_dec.item() * batch_size
        triplet_loss_total += triplet_loss.item() * len(anchor_idx)
        batch_count += 1

    if batch_count > 0:
        enc_loss_total /= (batch_count * args['batch_size'])
        dec_loss_total /= (batch_count * args['batch_size'])
        disc_loss_total /= (batch_count * args['batch_size'])
        mse_loss_total /= (batch_count * args['batch_size'])
        triplet_loss_total /= (batch_count * args['batch_size'])

    print(f"Epoch: {epoch} \tEnc Loss: {enc_loss_total:.6f} \tDec Loss: {dec_loss_total:.6f} \tDisc Loss: {disc_loss_total:.6f} \tMSE: {mse_loss_total:.6f} \tTriplet: {triplet_loss_total:.6f}")

    # Save losses to CSV
    loss_dict = {
        'enc_loss': enc_loss_total,
        'dec_loss': dec_loss_total,
        'disc_loss': disc_loss_total,
        'mse_loss': mse_loss_total,
        'triplet_loss': triplet_loss_total
    }
    save_losses(loss_dict, modpth, 'triplet_gan', epoch)

    if dec_loss_total < best_loss and args['save']:
        print('Saving..')
        torch.save(encoder.state_dict(), os.path.join(modpth, 'bst_enc.pth'))
        torch.save(decoder.state_dict(), os.path.join(modpth, 'bst_dec.pth'))
        torch.save(discriminator.state_dict(), os.path.join(modpth, 'bst_disc.pth'))
        best_loss = dec_loss_total

# Plot losses
plot_losses(os.path.join(modpth, 'triplet_gan_training_losses.csv'), modpth, 'triplet_gan')

t1 = time.time()
print(f"Total time (min): {(t1 - t0) / 60:.2f}")