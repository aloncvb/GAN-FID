from dcgan import Generator
import torch
from torchvision.models import inception_v3
import time
import numpy as np
import torch.nn as nn
import argparse
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
import torch, torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from diff_fid import inception_model, get_activation_statistics, frechet_distance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_inception_net():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    return inception_model


print("Loading inception... ", end="", flush=True)
t0 = time.time()
# inception_model = load_inception_net().cuda().eval().half()
print("DONE! %.4fs" % (time.time() - t0))

mu1 = np.load("I128_inception_moments.npz")["mu"]
sigma1 = np.load("I128_inception_moments.npz")["sigma"]

mu_gpu = torch.from_numpy(mu1).cuda().float()
sigma_gpu = torch.from_numpy(sigma1).cuda().float()


# START FASTFID CODE
def compute_trace(S):
    tr = torch.sum(torch.norm(S, dim=0) ** 2)
    return tr


def trace_of_matrix_sqrt(C1, C2):
    d, bs = C1.shape
    M = C1.t() @ C2 @ C1
    S = torch.svd(M.float(), compute_uv=True)[1].half()  # PSD => singular=eigen
    S = torch.topk(S, bs - 1)[0]
    return torch.sum(torch.sqrt(S))


def calculate_activation_statistics(images1, batch_size):
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
        act1[start_idx:end_idx, :] = inception_model(images1[start_idx:end_idx])[0]
    act1 = act1.t()

    d, bs = act1.shape
    all_ones = torch.ones((1, bs)).cuda()
    mu1 = torch.mean(act1, axis=1).view(d, 1)
    S1 = np.sqrt(1 / (bs - 1)) * (act1 - mu1 @ all_ones)
    return mu1, S1


def calculate_frechet_distance_fast(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1.view(2048) - mu2.view(2048)
    tr_covmean = trace_of_matrix_sqrt(sigma1, sigma2)
    mu = diff.dot(diff)
    tr1 = compute_trace(sigma1)
    tr2 = torch.trace(sigma2)
    return mu + tr1 + tr2 - 2 * tr_covmean


def fastprefid(images1, mu2, sigma2, batch_size=-1):
    up = nn.Upsample((299, 299), mode="bilinear")
    images1 = up(images1)

    torch.cuda.empty_cache()
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    torch.cuda.empty_cache()
    fid = calculate_frechet_distance_fast(mu1, sigma1, mu2, sigma2)
    return fid


# toggle grad only in abstract layers.
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

    for index, blocklist in enumerate(model.blocks):
        if index > 2:
            for block in blocklist:
                for p in block.parameters():
                    p.requires_grad = False


def train(generator: Generator, trainloader: DataLoader):
    optim = Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.99))

    total_loss_g = 0
    batch_idx = 0
    for batch, _ in trainloader:
        data = batch.to(device)
        optim.zero_grad()
        batch_size = data.size()[0]

        real_mu, real_sigma = get_activation_statistics(
            data,
            device=device,
        )
        fake_images_fid = generator(batch_size)  # 1000 for stable score
        # use fid for better training
        fake_mu, fake_sigma = get_activation_statistics(
            fake_images_fid,
            device=device,
        )
        fid_loss = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        # * loss_g # loss_g is there to scale loss in the range of generator loss
        loss_g = fid_loss
        loss_g.backward()

        total_loss_g += loss_g.item()
        optim.step()
        batch_idx += 1
    return total_loss_g / batch_idx


def test(
    generator: Generator,
    testloader: DataLoader,
    filename: str,
    epoch: int,
):
    generator.eval()  # set to inference mode
    with torch.no_grad():
        samples = generator(100)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + "epoch%d.png" % epoch,
        )

        total_loss_g = 0
        total_loss_d = 0
        batch_idx = 0
        for batch, _ in testloader:
            data = batch.to(device)
            batch_size = data.size()[0]

            real_mu, real_sigma = get_activation_statistics(
                data,
                device=device,
            )
            fake_images_fid = generator(batch_size)  # 1000 for stable score
            # use fid for better training
            fake_mu, fake_sigma = get_activation_statistics(
                fake_images_fid,
                device=device,
            )
            fid_loss = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
            # * loss_g # loss_g is there to scale loss in the range of generator loss
            loss_g = fid_loss

            total_loss_g += loss_g.item()
            batch_idx += 1
        print(
            "Epoch: {} Test set: Average loss_d: {:.4f}".format(
                epoch, total_loss_d / batch_idx
            )
        )
        print(
            "Epoch: {} Test set: Average loss_g: {:.4f}".format(
                epoch, total_loss_g / batch_idx
            )
        )
    return total_loss_g / batch_idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (64),
                interpolation=transforms.InterpolationMode.BICUBIC,  # size_that_worked = 64
            ),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./data/MNIST", train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.MNIST(
            root="./data/MNIST", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    elif args.dataset == "fashion-mnist":
        trainset = torchvision.datasets.FashionMNIST(
            root="~/torch/data/FashionMNIST",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data/FashionMNIST", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    else:
        raise ValueError("Dataset not implemented")

    filename = (
        "%s_" % args.dataset + "batch%d_" % args.batch_size + "mid%d_" % args.latent_dim
    )

    generator = Generator(100).to(device)
    generator.load_state_dict(
        torch.load("models/generator.pt", map_location=device), strict=False
    )

    loss_train_arr_g = []
    loss_test_arr_g = []
    for epoch in range(1, args.epochs + 1):
        loss_train_g = train(generator, trainloader)
        loss_train_arr_g.append(loss_train_g)
        loss_test_g = test(generator, testloader, filename, epoch)
        loss_test_arr_g.append(loss_test_g)
    # Save the model
    torch.save(generator.state_dict(), "generator.pt")

    # create a plot of the loss
    plt.plot(loss_train_arr_g, label="train_g")
    plt.plot(loss_test_arr_g, label="test_g")
    plt.xlabel("Epoch")
    plt.ylabel("fid loss")
    plt.legend()
    plt.savefig("fid_loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--dataset", help="dataset to be modeled.", type=str, default="mnist"
    )
    parser.add_argument(
        "--batch_size", help="number of images in a mini-batch.", type=int, default=64
    )
    parser.add_argument(
        "--epochs", help="maximum number of iterations.", type=int, default=20
    )
    parser.add_argument(
        "--sample_size", help="number of images to generate.", type=int, default=64
    )

    parser.add_argument("--latent-dim", help=".", type=int, default=100)
    parser.add_argument(
        "--lr", help="initial learning rate.", type=float, default=0.0002
    )

    args = parser.parse_args()
    main(args)
