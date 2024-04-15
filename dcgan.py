"""DCGAN model
"""

import torch
import torch.nn as nn


image_size = 28
nc = 1  # Number of channels in the training images. For color images this is 3
feature_num = 128  # Size of feature maps in generator/discriminator


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_num * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.ReLU(True),
            # State size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_num * 8, feature_num * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.ReLU(True),
            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_num * 4, feature_num * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.ReLU(True),
            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_num * 2, feature_num, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(True),
            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_num, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, feature_num, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf) x 32 x 32
            nn.Conv2d(feature_num, feature_num * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_num * 2, feature_num * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_num * 4, feature_num * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*8) x 4 x 4
            nn.Conv2d(feature_num * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class DCGAN:
    def __init__(self, latent_dim, device):
        super(DCGAN, self).__init__()
        self.generator = Generator(latent_dim)
        self.generator.to(device)
        self.discriminator = Discriminator()
        self.discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        self.loss = nn.BCELoss()

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def label(self, x):
        return self.discriminator(x)

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, 1, 1)

    def generate_fake(self, batch_size):
        return self.generator(self.generate_latent(batch_size))

    def label_real(self, images):
        return self.label(images)

    def label_fake(self, batch_size):
        return self.label(self.generate_fake(batch_size))

    def calculate_dicriminator_loss(self, real, fake, batch_size):
        real_label = torch.ones(batch_size, device=self.device)
        fake_label = torch.zeros(batch_size, device=self.device)
        return self.loss(real, real_label) + self.loss(fake, fake_label)

    def calculate_generator_loss(self, fake, batch_size):
        real_label = torch.ones(batch_size, device=self.device)
        return self.loss(fake, real_label)
