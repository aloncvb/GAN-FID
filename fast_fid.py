import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np


class FastFID(nn.Module):
    def __init__(self, device):
        super(FastFID, self).__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.to(device)
        self.inception.eval()

    def forward(self, real_imgs, fake_imgs):
        # Compute features from Inception model
        real_feats = self.inception(real_imgs)
        fake_feats = self.inception(fake_imgs)

        # Calculate means and covariance matrices
        mu_real, cov_real = self.compute_stats(real_feats)
        mu_fake, cov_fake = self.compute_stats(fake_feats)

        # Compute the squared norm of the difference in means
        mean_diff = torch.norm(mu_real - mu_fake, p=2) ** 2

        # Efficiently compute the trace of the square root of covariance product
        tr_sqrt_product = self.fast_trace_sqrt_product(cov_real, cov_fake)

        # FID formula as given in the paper
        fid_score = mean_diff + tr_sqrt_product
        return fid_score

    def compute_stats(self, features):
        mu = torch.mean(features, dim=0)
        features_centered = features - mu
        cov = torch.mm(features_centered.t(), features_centered) / (
            features_centered.size(0) - 1
        )
        return mu, cov

    def fast_trace_sqrt_product(self, cov_real, cov_fake):
        # Placeholder for fast trace sqrt computation of covariances
        # Real implementation would require computing the eigenvalues
        # and then their square roots, which is simplified here
        return torch.trace(torch.sqrt(cov_real + cov_fake)) - 2 * torch.trace(
            torch.sqrt(torch.mm(cov_real, cov_fake))
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
