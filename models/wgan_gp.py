# For WGAN-GP we can reuse the same architecture as WGAN above. Naming adapted.

import torch
import torch.nn as nn

class WGANGPGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_channels=3, feature_g=64):
        super(WGANGPGenerator, self).__init__()
        input_dim = latent_dim + num_classes
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

            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.deconv(x)


class WGANGPDiscriminator(nn.Module):
    def __init__(self, num_classes=10, img_channels=3, feature_d=64):
        super(WGANGPDiscriminator, self).__init__()
        in_channels = img_channels + num_classes
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, 1, 4, 2, 0),
        )

    def forward(self, img, labels_onehot):
        N, C, H, W = img.size()
        label_map = labels_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        out = self.disc(torch.cat([img, label_map], dim=1))
        return out.view(-1)
