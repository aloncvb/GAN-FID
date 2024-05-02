import torch
import numpy as np
from torchvision import transforms

import torch.utils
import torch.utils.data
import torch

import torch
from torchvision import transforms, datasets
from diff_fid import inception_feature_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inception_model = inception_feature_extractor()


def calculate_activation_statistics(images1, batch_size, inception=None):
    batch_size = int(batch_size.detach().cpu().numpy()[0])
    num_images = images1.shape[0]
    assert (
        num_images % batch_size == 0
    ), "Please choose batch_size to divide number of images."
    n_batches = int(num_images / batch_size)
    act1 = torch.zeros((num_images, 2048)).cuda()

    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)
        act1[start_idx:end_idx, :] = inception(images1[start_idx:end_idx])[0]
    act1 = act1.t()

    d, bs = act1.shape
    all_ones = torch.ones((1, bs)).cuda()
    mu1 = torch.mean(act1, axis=1).view(d, 1)
    S1 = np.sqrt(1 / (bs - 1)) * (act1 - mu1 @ all_ones)
    return mu1, S1


def calc_images_stats():
    # Load MNIST dataset
    dataset = datasets.CIFAR10(
        root="./data",
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(64),  # Resize to match Inception input
                transforms.ToTensor(),
            ]
        ),
    )

    mu, sigma = calculate_activation_statistics(
        dataset, 1024, inception=inception_model
    )
    mu = mu.cpu().numpy()  # Move to CPU and convert to NumPy array
    sigma = sigma.cpu().numpy()  # Move to CPU and convert to NumPy array

    # Save 'mu' and 'sigma' to a .npz file
    np.savez("cifar_inception.npz", mu=mu, sigma=sigma)


# Specify the directory to save the images
calc_images_stats()
