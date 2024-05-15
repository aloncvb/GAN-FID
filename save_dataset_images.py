"""
this file is used to save dataset images to later use for the FID score calculation
"""

import os
from torchvision import datasets, transforms


def save_mnist_images(directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o777)

    # Load MNIST dataset
    mnist = datasets.MNIST(
        root="./data",
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(299),  # Resize to match Inception input
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                transforms.ToTensor(),
            ]
        ),
    )

    # Save images
    for i, (image, _) in enumerate(mnist):
        # Convert tensor to PIL Image
        image = transforms.ToPILImage()(image)
        # Save image
        image.save(os.path.join(directory, f"mnist_{i}.png"))


# Specify the directory to save the images
save_directory = "mnist_images"
save_mnist_images(save_directory)
