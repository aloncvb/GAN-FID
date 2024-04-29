import torch
from torchvision.models import inception_v3
import torch.nn.functional as F
import torch.nn as nn


def inception_feature_extractor():
    model = inception_v3(pretrained=True, init_weights=False)
    model.fc = torch.nn.Identity()  # Replace the classifier with an identity function
    model.eval()  # Set the model to evaluation mode
    return model


inception_model = inception_feature_extractor()


def get_activation_statistics(images, device="cuda"):
    inception_model.to(device)
    inception_model.eval()
    up = nn.Upsample((299, 299), mode="bilinear")
    upsizing_images = up(images)

    with torch.no_grad():
        pred = inception_model(upsizing_images)

    mu = torch.mean(pred, dim=0)
    sigma = torch.cov(pred)
    return mu, sigma


def trace_of_matrix_sqrt(C1, C2):
    """
    Computes using the fact that:   eig(A @ B) = eig(B @ A)

    C1, C2    (d, bs)

    M = C1 @ C1.T @ C2 @ C2.T

    eig ( C1 @ C1.T @ C2 @ C2.T ) =
    eig ( C1 @ (C1.T @ C2) @ C2.T ) =      O(d bs^2)
    eig ( C1 @ ((C1.T @ C2) @ C2.T) ) =        O(d bs^2)
    eig ( ((C1.T @ C2) @ C2.T) @ C1 ) =        O(d bs^2)
    eig ( batch_size x batch_size  )      O(bs^3)

    """
    d, bs = C1.shape
    assert bs <= d, (
        "This algorithm takes O(bs^2d) time instead of O(d^3), so only use it when bs < d.\nGot bs=%i>d=%i. "
        % (bs, d)
    )  # it also computes wrong thing sice it returns bs eigenvalues and there are only d.
    M = ((C1.t() @ C2) @ C2.t()) @ C1  # computed in O(d bs^2) time.    O(d^^3)
    S = torch.svd(M, compute_uv=True)[1]  # need 'uv' for backprop.
    S = torch.topk(S, bs - 1)[0]  # covariance matrix has rank bs-1
    return torch.sum(torch.sqrt(S))


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> torch.Tensor:
    """Calculation of the Frechet Distance between two Gaussians."""
    mu1 = mu1.to(torch.float32).requires_grad_(True)
    mu2 = mu2.to(torch.float32)
    sigma1 = sigma1.to(torch.float32)
    sigma2 = sigma2.to(torch.float32)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = trace_of_matrix_sqrt(sigma1, sigma2)
    if not torch.isfinite(covmean).all():
        covmean = covmean + torch.eye(sigma1.size(0)) * eps

    # FID calculation
    dist = (diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * covmean
    return dist.requires_grad_(True)
