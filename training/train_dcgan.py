import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.dcgan import Generator, Discriminator
from utils.datasets import get_dataset
from utils.plotting import plot_loss_curve

# Hyperparameters
batch_size = 128
lr = 0.0002
epochs = 100
latent_dim = 100
dataset_name = "mnist"   # change as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
os.makedirs("results/loss_curves", exist_ok=True)
os.makedirs("checkpoints/dcgan", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize(64),   # DCGAN expects 64x64
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = get_dataset(dataset_name, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

g_losses, d_losses = [], []

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        real = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(imgs), real)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), "checkpoints/dcgan/generator.pth")
torch.save(discriminator.state_dict(), "checkpoints/dcgan/discriminator.pth")

# Save loss curve
plot_loss_curve(g_losses, d_losses, "results/loss_curves/dcgan_loss.png")
