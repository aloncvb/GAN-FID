import argparse
import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from diff_fid import get_activation_statistics, frechet_distance
from dcgan import Generator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    generator: Generator,
    trainloader: DataLoader,
    optim: Adam,
):
    generator.train()
    total_loss_g = 0
    batch_idx = 0
    for batch, _ in trainloader:
        data = batch.to(device)
        noise = torch.randn(data.size()[0], 100, 1, 1, device=device)
        optim.zero_grad()

        real_mu, real_sigma = get_activation_statistics(
            data,
            device=device,
        )
        fake_images_fid = generator(noise)
        # use fid for better training
        fake_mu, fake_sigma = get_activation_statistics(
            fake_images_fid,
            device=device,
        )
        fid_loss = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        fid_loss.backward()

        total_loss_g += fid_loss.item()
        optim.step()
        batch_idx += 1
    return total_loss_g / batch_idx


def test(
    generator: Generator, testloader: DataLoader, filename: str, epoch: int, fixed_noise
):
    generator.eval()  # set to inference mode
    with torch.no_grad():
        samples = generator(fixed_noise[:100])
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + "epoch%d.png" % epoch,
        )

        total_loss_g = 0
        batch_idx = 0
        for index, (batch, _) in enumerate(testloader):
            data = batch.to(device)
            batch_size = data.size()[0]

            real_mu, real_sigma = get_activation_statistics(
                data,
                device=device,
            )
            fake_images_fid = generator(
                fixed_noise[index * batch_size : (index + 1) * batch_size]
            )  # 1000 for stable score
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
            "Epoch: {} Test set: Average loss_g: {:.4f}".format(
                epoch, total_loss_g / batch_idx
            )
        )
    return total_loss_g / batch_idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
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
    elif args.dataset == "cifar":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (64),
                    interpolation=transforms.InterpolationMode.BICUBIC,  # size_that_worked = 64
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="~/torch/data/Cifar10",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data/Cifar10", train=False, download=True, transform=transform
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
    optim = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    loss_train_arr_g = []
    loss_test_arr_g = []
    fixed_noise = torch.randn(testset.__len__(), 100, 1, 1, device=device)
    for epoch in range(1, args.epochs + 1):
        loss_train_g = train(generator, trainloader, optim)
        loss_train_arr_g.append(loss_train_g)
        # print train loss:
        print("Epoch: {} Train set: Average loss_g: {:.4f}".format(epoch, loss_train_g))
        loss_test_g = test(generator, testloader, filename, epoch, fixed_noise)
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
        "--lr", help="initial learning rate.", type=float, default=0.00002
    )

    args = parser.parse_args()
    main(args)
