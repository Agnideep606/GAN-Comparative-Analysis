import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet32 import ResNet32
from utils.datasets import get_generated_dataset
from utils.metrics import accuracy, precision_recall_f1

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "cifar10"  # ["mnist", "fmnist", "cifar10", "svhn"]
GEN_PATH = f"generation/{DATASET}_generated"  # folder with generated images

# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if DATASET == "mnist":
    train_real = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
    test_real = datasets.MNIST(root="data/", train=False, download=True, transform=transform)
elif DATASET == "fmnist":
    train_real = datasets.FashionMNIST(root="data/", train=True, download=True, transform=transform)
    test_real = datasets.FashionMNIST(root="data/", train=False, download=True, transform=transform)
elif DATASET == "cifar10":
    train_real = datasets.CIFAR10(root="data/", train=True, download=True, transform=transform)
    test_real = datasets.CIFAR10(root="data/", train=False, download=True, transform=transform)
elif DATASET == "svhn":
    train_real = datasets.SVHN(root="data/", split="train", download=True, transform=transform)
    test_real = datasets.SVHN(root="data/", split="test", download=True, transform=transform)

train_generated = get_generated_dataset(GEN_PATH, transform=transform)

# Mix 50% real + 50% generated
train_dataset = ConcatDataset([train_real, train_generated])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ----------------------------
# Model
# ----------------------------
model = ResNet32(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/resnet32_{DATASET}.pth")
print("✅ ResNet-32 training finished and model saved.")
