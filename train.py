"""Training procedure for gan.
"""

import matplotlib.pyplot as plt
import argparse
import torch.utils
import torch.utils.data
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from dcgan import DCGAN

BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers


def train(
    dcgan: DCGAN,
    trainloader: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
):
    dcgan.train()  # set to training mode
    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0
    for batch, _ in trainloader:
        data = batch.to(dcgan.device)
        print(f"device: {dcgan.device}")
        optimizer_d.zero_grad()
        # discriminator train
        real_label = dcgan.label_real(data)
        fake_label = dcgan.label_fake(batch_size=batch.size)
        print("label success")
        loss_d = dcgan.calculate_dicriminator_loss(real_label, fake_label, batch.size)
        print("calc loss d success")
        loss_d.backward()
        print("loss d backwards")
        total_loss_d += loss_d.item()
        print("total loss d backwards")
        optimizer_d.step()
        print("optimizer d step success")

        # generator train
        optimizer_g.zero_grad()
        results = dcgan.label_fake(batch_size=batch.size)
        lossG = dcgan.calculate_generator_loss(results, batch_size=batch.size)
        print("calc loss g success")
        lossG.backward()
        total_loss_g += lossG.item()
        print("total loss g success")
        optimizer_g.step()

        batch_idx += 1
    return total_loss_d / batch_idx, total_loss_g / batch_idx


def test(vae, testloader, filename, epoch):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        samples = vae.sample(100).to(vae.device)
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + "epoch%d.png" % epoch,
        )

        total_loss = 0
        batch_idx = 0
        for batch, _ in testloader:
            data = batch.to(vae.device)

            loss = vae(data)
            total_loss += loss.item()
            batch_idx += 1
        print(
            "Epoch: {} Test set: Average loss: {:.4f}".format(
                epoch, total_loss / batch_idx
            )
        )
    return total_loss / batch_idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + torch.zeros_like(x).uniform_(0.0, 1.0 / 256.0)
            ),  # dequantization
            transforms.Normalize((0.0,), (257.0 / 256.0,)),  # rescales to [0,1]
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

    dcgan = DCGAN(latent_dim=args.latent_dim, device=device)
    optimizer_d = torch.optim.Adam(
        dcgan.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_g = torch.optim.Adam(
        dcgan.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    loss_train_arr = []
    loss_test_arr = []
    for epoch in range(1, args.epochs + 1):
        loss_train = train(
            dcgan, trainloader, optimizer_d=optimizer_d, optimizer_g=optimizer_g
        )
        loss_train_arr.append(loss_train)
        loss_test = test(dcgan, testloader, filename, epoch)
        loss_test_arr.append(loss_test)
    # Save the model
    torch.save(dcgan.generator.state_dict(), "generator.pt")
    torch.save(dcgan.discriminator.state_dict(), "discriminator.pt")
    # create a plot of the loss
    plt.plot(loss_train_arr, label="train")
    plt.plot(loss_test_arr, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig("elbo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--dataset", help="dataset to be modeled.", type=str, default="mnist"
    )
    parser.add_argument(
        "--batch_size", help="number of images in a mini-batch.", type=int, default=128
    )
    parser.add_argument(
        "--epochs", help="maximum number of iterations.", type=int, default=50
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
