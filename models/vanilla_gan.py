import torch
import torch.nn as nn

# Vanilla conditional GAN for MNIST/FashionMNIST (28x28) or CIFAR after resizing
class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super(VanillaGenerator, self).__init__()
        self.img_shape = img_shape
        input_dim = latent_dim + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img


class VanillaDiscriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super(VanillaDiscriminator, self).__init__()
        self.img_shape = img_shape
        input_dim = int(torch.prod(torch.tensor(img_shape))) + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels_onehot):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat([img_flat, labels_onehot], dim=1)
        validity = self.model(x)
        return validity
