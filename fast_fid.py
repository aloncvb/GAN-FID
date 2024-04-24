import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np


class FastFID(nn.Module):
    def __init__(self, device):
        super(FastFID, self).__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity().to(device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((299, 299))
        self.inception.to(device)
        self.inception.eval()
        self.device = device

    def forward(self, real_images, fake_images):
        # Compute features from Inception model
        real_images = self.adaptive_pool(
            real_images
        )  # size is [batch_size, 3, 299, 299]
        fake_images = self.adaptive_pool(
            fake_images
        )  # size is [batch_size, 3, 299, 299]

        # Compute features from Inception model
        real_feats = self.inception(real_images).view(
            real_images.size(0), -1
        )  # size is [batch_size, 2048]
        fake_feats = self.inception(fake_images).view(
            fake_images.size(0), -1
        )  # size is [batch_size, 2048]

        # Calculate means and covariance matrices
        mu_real, cov_real = self.compute_stats(
            real_feats
        )  # mu_real size is [2048], cov real size is [2048, 2048]
        mu_fake, cov_fake = self.compute_stats(
            fake_feats
        )  # mu_fake size is [2048], cov fake size is [2048, 2048]

        # Compute the squared norm of the difference in means
        mean_diff = torch.norm(mu_real - mu_fake, p=2) ** 2  # size is []. why?
        # Efficiently compute the trace of the square root of covariance product
        tr_sqrt_product = self.fast_trace_sqrt_product(cov_real, cov_fake)
        if torch.isnan(
            tr_sqrt_product
        ).any():  # check for nan values in the trace sqrt product
            raise ValueError("Fast FID has returned nan values")
        # FID formula as given in the paper
        fid_score = mean_diff + tr_sqrt_product
        return fid_score

    def compute_stats(self, features):
        # Assume features is of shape [batch_size, num_features]
        mu = torch.mean(features, dim=0)
        features_centered = features - mu
        cov = torch.mm(features_centered.t(), features_centered) / (
            features_centered.size(0) - 1
        )
        # Regularization to ensure the covariance matrix is positive semidefinite
        epsilon = 1e-3
        cov += torch.eye(cov.size(0), device=cov.device) * epsilon

        return mu, cov

    def fast_trace_sqrt_product(self, cov_real, cov_fake):
        # Placeholder for fast trace sqrt computation of covariances
        # Real implementation would require computing the eigenvalues
        # and then their square roots, which is simplified here
        # return torch.trace(torch.sqrt(cov_real + cov_fake)) - 2 * torch.trace(
        #     torch.sqrt(torch.mm(cov_real, cov_fake))
        # )

        # # Use Cholesky decomposition to find the square roots of the matrices
        # sqrt_cov_real = torch.linalg.cholesky(cov_real)
        # sqrt_cov_fake = torch.linalg.cholesky(cov_fake)

        # # Compute the product of square roots and then its square root
        # product_sqrt = torch.mm(sqrt_cov_real, sqrt_cov_fake)
        # sqrt_product = torch.linalg.cholesky(product_sqrt)

        # # Trace of the final product
        # tr_sqrt_product = torch.trace(sqrt_product)
        # return tr_sqrt_product
        U_real, S_real, V_real = torch.linalg.svd(cov_real)
        sqrt_cov_real = torch.mm(U_real, torch.diag(torch.sqrt(S_real)))

        U_fake, S_fake, V_fake = torch.linalg.svd(cov_fake)
        sqrt_cov_fake = torch.mm(U_fake, torch.diag(torch.sqrt(S_fake)))

        # Compute the product of square roots and then its square root using SVD again
        product_sqrt = torch.mm(sqrt_cov_real, sqrt_cov_fake)
        U_prod, S_prod, V_prod = torch.linalg.svd(product_sqrt)
        sqrt_product = torch.mm(U_prod, torch.diag(torch.sqrt(S_prod)))

        # Trace of the final product
        tr_sqrt_product = torch.sum(S_prod)
        return tr_sqrt_product
