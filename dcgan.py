"""DCGAN model
"""

import torch
import torch.nn as nn


image_size = 28
nc = 3  # Number of channels in the training images. For color images this is 3
feature_num = 128  # Size of feature maps in generator/discriminator


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_num * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_num * 8),
            nn.ReLU(True),
            # State size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_num * 8, feature_num * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 4),
            nn.ReLU(True),
            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_num * 4, feature_num * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 2),
            nn.ReLU(True),
            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_num * 2, feature_num, 4, 2, 1),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(True),
            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_num, nc, 4, 2, 1),
            nn.Tanh(),
            # Output size. (nc) x 299 x 299
        )

    def forward(self, input):
        output = self.main(input)
        # print(
        #     "Generator output size:", output.size()
        # )  # Should be [batch_size, 3, 299, 299] if correctly configured
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 128
            nn.Conv2d(nc, feature_num, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 2, feature_num * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 4, feature_num * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output


class DCGAN:
    def __init__(self, latent_dim, device):
        super(DCGAN, self).__init__()
        self.generator = Generator(latent_dim)
        self.generator.to(device)
        self.discriminator = Discriminator()
        self.discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        self.loss = nn.BCELoss()  # change to mse loss for bce loss

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def label(self, x):
        return self.discriminator(x).squeeze()

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device).view(
            -1, self.latent_dim, 1, 1
        )

    def generate_fake(self, batch_size):
        return self.generator(self.generate_latent(batch_size))

    def label_real(self, images):
        return self.label(images)

    def label_fake(self, batch_size):
        return self.label(self.generate_fake(batch_size))

    def calculate_dicriminator_loss(self, real, fake, batch_size):
        soft_real = torch.full(real.size(), 0.9, device=self.device)
        soft_fake = torch.full(fake.size(), 0.1, device=self.device)
        # real_label = torch.ones(batch_size, device=self.device)
        # fake_label = torch.zeros(batch_size, device=self.device)
        return self.loss(real, soft_real) + self.loss(fake, soft_fake)

    def calculate_generator_loss(self, dis_label, batch_size):
        # real_label = torch.ones(batch_size, device=self.device)
        soft_real = torch.full(dis_label.size(), 0.9, device=self.device)
        return self.loss(dis_label, soft_real)
