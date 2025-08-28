import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet32 import ResNet32
from utils.metrics import accuracy, precision_recall_f1

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "cifar10"  # ["mnist", "fmnist", "cifar10", "svhn"]
MODEL_PATH = f"checkpoints/resnet32_{DATASET}.pth"

# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if DATASET == "mnist":
    test_set = datasets.MNIST(root="data/", train=False, download=True, transform=transform)
elif DATASET == "fmnist":
    test_set = datasets.FashionMNIST(root="data/", train=False, download=True, transform=transform)
elif DATASET == "cifar10":
    test_set = datasets.CIFAR10(root="data/", train=False, download=True, transform=transform)
elif DATASET == "svhn":
    test_set = datasets.SVHN(root="data/", split="test", download=True, transform=transform)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ----------------------------
# Load Model
# ----------------------------
model = ResNet32(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# Evaluation
# ----------------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy(all_labels, all_preds)
prec, rec, f1 = precision_recall_f1(all_labels, all_preds)

print(f"✅ Evaluation on {DATASET} Test Set")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
