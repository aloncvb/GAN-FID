# this file is used to calculate the FID score and the inception score
import torch
from torch.nn.functional import softmax
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_gan_metrics import get_inception_score
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy
from torchvision import transforms
import os
from PIL import Image


def inception_score(imgs, batch_size=128, splits=1):
    """Computes the inception score of the generated images."""
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    print("Loading Inception model")
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(
        device
    )
    print("Inception model loaded")
    inception_model.eval()
    print("Inception model set to evaluation mode")

    def get_pred(x):
        try:
            print("inception model(x)")
            x = inception_model(x)
            print("softmax(x, dim=1).data.numpy()")
            return softmax(x, dim=1).data.cpu().numpy()

        except Exception as e:
            print(e)
            raise e

    print("Getting predictions")
    # Get predictions
    preds = np.zeros((N, 1000))
    try:
        print("Calculating predictions")
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            batch_size_i = batch.size()[0]
            preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batch)
    except Exception as e:
        print(e)
        raise e

    try:
        # Now compute the mean kl-div
        split_scores = []
        print("Calculating inception score")
        for k in range(splits):
            part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        print("Inception score calculated")
        return np.mean(split_scores), np.std(split_scores)
    except Exception as e:
        print(e)
        raise e


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Path to the image directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_filenames = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = ["mnist_images", "generated_mnist_images"]

    # Calculate FID Score
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),  # Convert image to tensor
    #     ]
    # )
    # # Create a dataset from the image directory
    # directory = "./generated_mnist_images"
    # images = [
    #     os.path.join(directory, f)
    #     for f in os.listdir(directory)
    #     if os.path.isfile(os.path.join(directory, f))
    # ]
    # # str to pil image:

    # images = [transform(Image.open(img)) for img in images]
    # dataset = ImageDataset(directory="./generated_mnist_images", transform=transform)

    # # Create a DataLoader
    # dataloader = DataLoader(dataset, batch_size=128)
    # print("Calculating Inception Score")
    # is_score = get_inception_score(dataloader, use_torch=True)

    # print(f"Inception Score: {is_score}")

    # fid score
    print("Calculating FID Score")
    fid_value = calculate_fid_given_paths(
        paths, batch_size=128, device=device, dims=2048
    )

    print(f"FID Score: {fid_value}")
