import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter as P
from torch.utils.checkpoint import checkpoint
from torchvision.models import inception_v3, Inception_V3_Weights


class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = P(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        def inception_wrap(x):
            with torch.cuda.amp.autocast():
                x = (x + 1.0) / 2.0
                x = (x - self.mean) / self.std
                if x.shape[2] != 299 or x.shape[3] != 299:
                    x = F.interpolate(
                        x, size=(299, 299), mode="bilinear", align_corners=True
                    )
                x = self.net.Conv2d_1a_3x3(x)
                x = self.net.Conv2d_2a_3x3(x)
                x = self.net.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.net.Conv2d_3b_1x1(x)
                x = self.net.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.net.Mixed_5b(x)
                x = self.net.Mixed_5c(x)

                x = self.net.Mixed_5d(x)
                x = self.net.Mixed_6a(x)
                x = self.net.Mixed_6b(x)
                x = self.net.Mixed_6c(x)
                x = self.net.Mixed_6d(x)
                x = self.net.Mixed_6e(x)
                x = self.net.Mixed_7a(x)
                x = self.net.Mixed_7b(x)
                x = self.net.Mixed_7c(x)
                return x

        x = checkpoint(inception_wrap, x, use_reentrant=False)

        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        return pool, logits


def inception_feature_extractor(half=True) -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model = WrapInception(model)
    if half:
        return model.eval().half()
    return model.eval()


inception_model = inception_feature_extractor()


def get_activation_statistics(images, device="cuda"):
    inception_model.to(device)
    up = nn.Upsample((299, 299), mode="bilinear")
    upsizing_images = up(images)

    with torch.no_grad():
        pred = inception_model(upsizing_images)[0]

    mu = torch.mean(pred, dim=0)
    sigma = torch.cov(pred)
    return mu, sigma


def trace_of_matrix_sqrt(C1, C2):
    """
    Computes using the fact that:   eig(A @ B) = eig(B @ A)
    """
    d, bs = C1.shape
    assert bs <= d, "error at trace of matrix"
    M = ((C1.t() @ C2) @ C2.t()) @ C1
    S = torch.svd(M, compute_uv=True)[1]  # need 'uv' for backprop.
    S = torch.topk(S, bs - 1)[0]  # covariance matrix has rank bs-1
    return torch.sum(torch.sqrt(S))


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> torch.Tensor:
    """Calculation of the Frechet Distance between two Gaussians."""
    mu1 = mu1.to(torch.float32)
    mu2 = mu2.to(torch.float32)
    sigma1 = sigma1.to(torch.float32)
    sigma2 = sigma2.to(torch.float32)

    diff = mu1 - mu2

    covmean = trace_of_matrix_sqrt(sigma1, sigma2)
    if not torch.isfinite(covmean).all():
        covmean = covmean + torch.eye(sigma1.size(0)) * eps

    # FID calculation
    dist = (diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * covmean
    return dist.requires_grad_(True)
