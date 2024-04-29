import torch
from torchvision.models import inception_v3
import torch.nn.functional as F


def inception_feature_extractor():
    model = inception_v3(pretrained=True, init_weights=False)
    model.fc = torch.nn.Identity()  # Replace the classifier with an identity function
    model.eval()  # Set the model to evaluation mode
    return model


inception_model = inception_feature_extractor()


def get_activation_statistics(images, model, batch_size=50, dims=2048, device="cuda"):
    model.to(device)
    model.eval()
    upsizing_images = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )
    n_batches = len(upsizing_images) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = torch.empty((n_used_imgs, dims)).to(device)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = upsizing_images[start:end].to(device)
        with torch.no_grad():
            pred = model(batch)

        pred_arr[start:end] = pred.view(batch.size(0), -1)

    mu = torch.mean(pred_arr, dim=0)
    sigma = torch.cov(pred_arr)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> torch.Tensor:
    """Calculation of the Frechet Distance between two Gaussians."""
    mu1 = mu1.to(torch.float32)
    mu2 = mu2.to(torch.float32)
    sigma1 = sigma1.to(torch.float32)
    sigma2 = sigma2.to(torch.float32)

    diff = mu1 - mu2
    # Product might be almost singular
    covmean = torch.sqrt(torch.matmul(sigma1, sigma2))

    # Numerically stabilize with epsilon
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
