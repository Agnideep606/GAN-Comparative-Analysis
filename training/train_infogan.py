import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.infogan import Generator, Discriminator, QNetwork
from utils.datasets import get_dataset
from utils.plotting import plot_loss_curve

# -------------------
# Hyperparameters
# -------------------
batch_size = 128
lr = 0.0002
epochs = 100
latent_dim = 62          # noise z
categorical_dim = 10     # categorical latent code (discrete)
continuous_dim = 2       # continuous latent codes
dataset_name = "mnist"   # dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
os.makedirs("results/loss_curves", exist_ok=True)
os.makedirs("checkpoints/infogan", exist_ok=True)

# Dataset
train_dataset = get_dataset(dataset_name, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(latent_dim, categorical_dim, continuous_dim).to(device)
discriminator = Discriminator().to(device)
q_network = QNetwork().to(device)

criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

optimizer_D = optim.Adam(list(discriminator.parameters()) + list(q_network.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_G = optim.Adam(list(generator.parameters()) + list(q_network.parameters()), lr=lr, betas=(0.5, 0.999))

g_losses, d_losses = [], []

# -------------------
# Training Loop
# -------------------
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        batch_size_curr = imgs.size(0)

        # Real / Fake labels
        real = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # -----------------
        # Sample latent codes
        # -----------------
        z = torch.randn(batch_size_curr, latent_dim, device=device)

        # categorical code (one-hot)
        categorical_code = torch.randint(0, categorical_dim, (batch_size_curr,), device=device)
        categorical_onehot = torch.zeros(batch_size_curr, categorical_dim, device=device)
        categorical_onehot[torch.arange(batch_size_curr), categorical_code] = 1.0

        # continuous code
        continuous_code = torch.rand(batch_size_curr, continuous_dim, device=device) * 2 - 1  # range [-1, 1]

        # full input
        latent_input = torch.cat([z, categorical_onehot, continuous_code], dim=1)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()
        gen_imgs = generator(latent_input)

        validity = discriminator(gen_imgs)
        g_loss = criterion_bce(validity, real)

        # Info Loss
        q_logits, q_mu, q_var = q_network(gen_imgs)
        info_loss_cat = criterion_ce(q_logits, categorical_code)
        info_loss_cont = criterion_mse(q_mu, continuous_code)
        info_loss = info_loss_cat + info_loss_cont

        g_total_loss = g_loss + 0.1 * info_loss
        g_total_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_loss = criterion_bce(discriminator(imgs), real)
        fake_loss = criterion_bce(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    g_losses.append(g_total_loss.item())
    d_losses.append(d_loss.item())
    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_total_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), "checkpoints/infogan/generator.pth")
torch.save(discriminator.state_dict(), "checkpoints/infogan/discriminator.pth")
torch.save(q_network.state_dict(), "checkpoints/infogan/q_network.pth")

# Save loss curve
plot_loss_curve(g_losses, d_losses, "results/loss_curves/infogan_loss.png")
