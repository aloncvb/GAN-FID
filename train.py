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
from fast_fid import FastFID

gradient_clip = 1e-1


def train(
    dcgan: DCGAN,
    trainloader: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
    fast_fid: FastFID,
):
    dcgan.train()  # set to training mode

    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0
    for batch, _ in trainloader:
        data = batch.to(dcgan.device)
        optimizer_d.zero_grad()
        batch_size = data.size()[0]
        # discriminator train
        real_label = dcgan.label_real(data)
        fake_label = dcgan.label_fake(batch_size=batch_size)
        loss_d = dcgan.calculate_dicriminator_loss(real_label, fake_label)
        loss_d.backward()
        for param in dcgan.discriminator.parameters():
            param.grad.data.clamp_(-gradient_clip, gradient_clip)
        total_loss_d += loss_d.item()
        optimizer_d.step()

        # generator train
        optimizer_g.zero_grad()
        fake_images = dcgan.generate_fake(batch_size)
        results = dcgan.label(fake_images)
        loss_g = dcgan.calculate_generator_loss(results)
        # use fid for better training
        fid_loss: torch.Tensor = fast_fid(
            real_images=data, fake_images=fake_images
        )  # Differentiable FID loss
        limit_loss = loss_g.item() * (1e-5 * fid_loss.item())
        loss_g = loss_g + limit_loss
        loss_g.backward()
        for param in dcgan.generator.parameters():
            param.grad.data.clamp_(-gradient_clip, gradient_clip)

        total_loss_g += loss_g.item()
        optimizer_g.step()
        batch_idx += 1
    return total_loss_d / batch_idx, total_loss_g / batch_idx


def test(
    dcgan: DCGAN, testloader: DataLoader, filename: str, epoch: int, fast_fid: FastFID
):
    dcgan.eval()  # set to inference mode
    with torch.no_grad():
        samples = dcgan.generate_fake(100)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + "epoch%d.png" % epoch,
        )

        total_loss_g = 0
        total_loss_d = 0
        batch_idx = 0
        for batch, _ in testloader:
            data = batch.to(dcgan.device)
            batch_size = data.size()[0]
            real_label = dcgan.label_real(data)
            fake_label = dcgan.label_fake(batch_size=batch_size)
            loss_d = dcgan.calculate_dicriminator_loss(
                real_label, fake_label, batch_size=batch_size
            )
            total_loss_d += loss_d.item()
            fake_images = dcgan.generate_fake(batch_size)
            results = dcgan.label(fake_images)
            loss_g = dcgan.calculate_generator_loss(results)

            # use fid for better training
            fid_loss = fast_fid(
                real_images=batch, fake_images=fake_images
            )  # Differentiable FID loss
            loss_g += fid_loss / 100

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
    return total_loss_d / batch_idx, total_loss_g / batch_idx


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

    dcgan = DCGAN(latent_dim=args.latent_dim, device=device)
    fast_fid = FastFID(device=device)
    optimizer_d = torch.optim.Adam(
        dcgan.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_g = torch.optim.Adam(
        dcgan.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    loss_train_arr_d = []
    loss_test_arr_d = []
    loss_train_arr_g = []
    loss_test_arr_g = []
    for epoch in range(1, args.epochs + 1):
        loss_train_d, loss_train_g = train(
            dcgan,
            trainloader,
            optimizer_d=optimizer_d,
            optimizer_g=optimizer_g,
            fast_fid=fast_fid,
        )
        loss_train_arr_d.append(loss_train_d)
        loss_train_arr_g.append(loss_train_g)
        loss_test_d, loss_test_g = test(
            dcgan, testloader, filename, epoch, fast_fid=fast_fid
        )
        loss_test_arr_d.append(loss_test_d)
        loss_test_arr_g.append(loss_test_g)
    # Save the model
    torch.save(dcgan.generator.state_dict(), "generator.pt")
    torch.save(dcgan.discriminator.state_dict(), "discriminator.pt")
    # create a plot of the loss
    plt.plot(loss_train_arr_d, label="train_d")
    plt.plot(loss_test_arr_d, label="test_d")
    plt.plot(loss_train_arr_g, label="train_g")
    plt.plot(loss_test_arr_g, label="test_g")
    plt.xlabel("Epoch")
    plt.ylabel("dcgan loss")
    plt.legend()
    plt.savefig("loss.png")


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
