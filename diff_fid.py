import torch
from torchvision.models import inception_v3
import torch.nn.functional as F


def inception_feature_extractor():
    model = inception_v3(pretrained=True, init_weights=False)
    model.fc = torch.nn.Identity()  # Replace the classifier with an identity function
    model.eval()  # Set the model to evaluation mode
    return model


inception_model = inception_feature_extractor()


def get_activation_statistics(images, device="cuda"):
    inception_model.to(device)
    inception_model.eval()
    upsizing_images = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        pred = inception_model(upsizing_images)

    pred = pred.detach().cpu().numpy()

    mu = torch.mean(pred, dim=0)
    sigma = torch.cov(pred)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> torch.Tensor:
    """Calculation of the Frechet Distance between two Gaussians."""
    mu1 = mu1.to(torch.float16)
    mu2 = mu2.to(torch.float16)
    sigma1 = sigma1.to(torch.float16)
    sigma2 = sigma2.to(torch.float16)

    diff = mu1 - mu2
    # Compute the product of covariance matrices
    prod = sigma1 @ sigma2

    # Compute the square root of the product matrix using SVD
    U, S, V = torch.svd(prod)
    covmean = (
        U @ torch.diag(torch.sqrt(S)) @ V.t()
    )  # Numerically stabilize with epsilon
    if not torch.isfinite(covmean).all():
        covmean = covmean + torch.eye(sigma1.size(0)) * eps

    # FID calculation
    dist = (
        (diff @ diff)
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2 * torch.trace(covmean)
    )
    return dist
