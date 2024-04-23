# this file is used to calculate the FID score and the inception score
import torch
from torch.nn.functional import softmax
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths
from scipy.stats import entropy
from torchvision import transforms, datasets
import os
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
paths = ["mnist_images", "generated_mnist_images"]


def inception_score(imgs, batch_size=128, resize=False, splits=1):
    """Computes the inception score of the generated images."""
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    print("Loading Inception model")
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    print("Inception model loaded")
    inception_model.eval()

    def get_pred(x):
        x = inception_model(x)
        return softmax(x, dim=1).data.numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    # Calculate FID Score

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert image to tensor
        ]
    )

    print("Calculating FID Score")
    fid_value = calculate_fid_given_paths(
        paths, batch_size=128, device=device, dims=2048
    )

    print(f"FID Score: {fid_value}")

    # Create a dataset from the image directory
    # dataset = datasets.ImageFolder(root="./generated_mnist_images", transform=transform)
    directory = "./generated_mnist_images"
    images = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    # str to pil image:

    images = [transform(Image.open(img)) for img in images]

    is_score = inception_score(images, batch_size=128, resize=True)

    print(f"Inception Score: {is_score}")
