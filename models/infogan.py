import torch
import torch.nn as nn

class InfoGANGenerator(nn.Module):
    def __init__(self, latent_dim=62, code_dim=2, img_shape=(1, 28, 28)):
        super(InfoGANGenerator, self).__init__()
        self.img_shape = img_shape
        input_dim = latent_dim + code_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img


class InfoGANDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(InfoGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disc = nn.Linear(256, 1)
        self.q = nn.Linear(256, 2)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        features = self.model(img_flat)
        validity = torch.sigmoid(self.disc(features))
        code = self.q(features)
        return validity, code
