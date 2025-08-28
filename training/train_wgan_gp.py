import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.wgan_gp import Generator, Discriminator
from utils.datasets import get_dataset
from utils.plotting import plot_loss_curve

# Hyperparameters
batch_size = 64
lr = 0.0001
epochs = 100
latent_dim = 100
lambda_gp = 10
n_critic = 5
dataset_name = "mnist"   # change as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
os.makedirs("results/loss_curves", exist_ok=True)
os.makedirs("checkpoints/wgan_gp", exist_ok=True)

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

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

g_losses, d_losses = [], []

def gradient_penalty(discriminator, real_imgs, fake_imgs):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_imgs.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
        real_loss = -torch.mean(discriminator(imgs))
        fake_loss = torch.mean(discriminator(gen_imgs.detach()))
        gp = gradient_penalty(discriminator, imgs.data, gen_imgs.data)
        d_loss = real_loss + fake_loss + lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()

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
torch.save(generator.state_dict(), "checkpoints/wgan_gp/generator.pth")
torch.save(discriminator.state_dict(), "checkpoints/wgan_gp/discriminator.pth")

# Save loss curve
plot_loss_curve(g_losses, d_losses, "results/loss_curves/wgan_gp_loss.png")
