import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.wgan import Generator, Discriminator
from utils.datasets import get_dataset
from utils.plotting import plot_loss_curve

# Hyperparameters
batch_size = 64
lr = 0.00005
epochs = 100
latent_dim = 100
clip_value = 0.01
n_critic = 5
dataset_name = "mnist"   # change as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
os.makedirs("results/loss_curves", exist_ok=True)
os.makedirs("checkpoints/wgan", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = get_dataset(dataset_name, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

g_losses, d_losses = [], []

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z).detach()
        d_loss = -torch.mean(discriminator(imgs)) + torch.mean(discriminator(gen_imgs))
        d_loss.backward()
        optimizer_D.step()

        # Weight clipping
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = -torch.mean(discriminator(gen_imgs))
            g_loss.backward()
            optimizer_G.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), "checkpoints/wgan/generator.pth")
torch.save(discriminator.state_dict(), "checkpoints/wgan/discriminator.pth")

# Save loss curve
plot_loss_curve(g_losses, d_losses, "results/loss_curves/wgan_loss.png")
