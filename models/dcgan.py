import torch
import torch.nn as nn

# DCGAN conditional generator/discriminator.
# For conditional input we concatenate label channels (one-hot expanded) to image channels for the discriminator,
# and concatenate one-hot to noise vector for generator.

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_channels=3, feature_g=64):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        input_dim = latent_dim + num_classes  # z + label one-hot

        # We'll map input vector to a small spatial feature map and apply ConvTranspose layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_g * 8 * 4 * 4),
            nn.BatchNorm1d(feature_g * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        # z: (N, latent_dim); labels_onehot: (N, num_classes)
        x = torch.cat([z, labels_onehot], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)  # (N, feature_g*8, 4, 4)
        img = self.deconv(x)
        return img


class DCGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10, img_channels=3, feature_d=64):
        super(DCGANDiscriminator, self).__init__()
        # discriminator will receive image channels + num_classes label channels (one-hot tiled spatially)
        in_channels = img_channels + num_classes
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels_onehot):
        # labels_onehot: (N, num_classes) -> expand to (N, num_classes, H, W)
        N, C, H, W = img.size()
        label_map = labels_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        x = torch.cat([img, label_map], dim=1)
        out = self.disc(x)
        return out.view(-1, 1).squeeze(1)
