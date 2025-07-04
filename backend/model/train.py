import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import DualInputSuperResDataset
from utils import PerceptualLoss, DualInputConsistencyLoss
from model import DualInputDiscriminator, DualInputESRGANGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model instances
generator = DualInputESRGANGenerator().to(device)
discriminator = DualInputDiscriminator().to(device)

# Checkpoint paths
gen_ckpt_path = "./models/dual_input_esrgan_generator.pth"
disc_ckpt_path = "./models/dual_input_esrgan_discriminator.pth"

# Ask user for loading weights or training from scratch
print("Model checkpoint management:")
if os.path.exists(gen_ckpt_path) and os.path.exists(disc_ckpt_path):
    choice = input("Saved model weights found. Do you want to load them? (y/n): ").strip().lower()
    if choice == 'y':
        generator.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_ckpt_path, map_location=device))
        print("Loaded existing model weights.")
    else:
        confirm = input("Training from scratch will DELETE existing weights. Have you backed them up? (y/n): ").strip().lower()
        if confirm == 'y':
            os.remove(gen_ckpt_path)
            os.remove(disc_ckpt_path)
            print("Old weights deleted. Starting from scratch.")
        else:
            print("Aborting. Please backup weights first.")
            exit(0)
else:
    print("No existing weights found. Training from scratch.")

# Loss functions
criterion_GAN = nn.BCELoss()
criterion_content = nn.L1Loss()
criterion_perceptual = PerceptualLoss().to(device)
criterion_consistency = DualInputConsistencyLoss().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

# Learning rate schedulers
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

# Dataset and DataLoader
dataset = DualInputSuperResDataset('data/train_hr', patch_size=128, scale_factor=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

print(f"\nDataset size: {len(dataset)}")
print(f"Batch size: 4")
print(f"Number of batches: {len(dataloader)}")

def train_model(num_epochs=100):
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for lr1, lr2, hr in progress_bar:
            lr1, lr2, hr = lr1.to(device), lr2.to(device), hr.to(device)
            batch_size = lr1.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_hr = generator(lr1, lr2)
            loss_GAN = criterion_GAN(discriminator(fake_hr), real_labels)
            loss_content = criterion_content(fake_hr, hr)
            loss_perceptual = criterion_perceptual(fake_hr, hr)
            loss_consistency = criterion_consistency(fake_hr, lr1, lr2)

            loss_G = loss_GAN + 100 * loss_content + 10 * loss_perceptual + 5 * loss_consistency
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            loss_D_real = criterion_GAN(discriminator(hr), real_labels)
            loss_D_fake = criterion_GAN(discriminator(fake_hr.detach()), fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            progress_bar.set_postfix({'G_loss': f'{loss_G.item():.4f}', 'D_loss': f'{loss_D.item():.4f}'})

        scheduler_G.step()
        scheduler_D.step()

        g_losses.append(epoch_g_loss / len(dataloader))
        d_losses.append(epoch_d_loss / len(dataloader))

        print(f"Epoch {epoch+1} - G_loss: {g_losses[-1]:.4f}, D_loss: {d_losses[-1]:.4f}")

        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                generator.eval()
                sample_lr1, sample_lr2, sample_hr = next(iter(dataloader))
                sample_lr1 = sample_lr1[:4].to(device)
                sample_lr2 = sample_lr2[:4].to(device)
                sample_hr = sample_hr[:4].to(device)
                fake_hr = generator(sample_lr1, sample_lr2)
                comparison = torch.cat([
                    F.interpolate(sample_lr1, scale_factor=4, mode='bicubic', align_corners=False),
                    F.interpolate(sample_lr2, scale_factor=4, mode='bicubic', align_corners=False),
                    fake_hr,
                    sample_hr
                ], dim=3)
                os.makedirs('outputs', exist_ok=True)
                vutils.save_image(comparison, f'outputs/epoch_{epoch+1}.png', normalize=True)
                generator.train()

    return g_losses, d_losses


if __name__ == "__main__":
    print("\nStarting dual-input super-resolution training...")
    NUM_EPOCHS = int(input("Enter number of epochs (e.g. 50): ").strip() or 50)

    g_losses, d_losses = train_model(NUM_EPOCHS)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.show()

    # Save model weights
    os.makedirs(os.path.dirname(gen_ckpt_path), exist_ok=True)
    torch.save(generator.state_dict(), gen_ckpt_path)
    torch.save(discriminator.state_dict(), disc_ckpt_path)

    print("\nTraining completed!")
    print(f"Models saved at:\n- {gen_ckpt_path}\n- {disc_ckpt_path}")
