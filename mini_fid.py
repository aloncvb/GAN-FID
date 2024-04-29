from dcgan import Generator
import torch
from torchvision import transforms
from torchvision.models import inception_v3
import time
import numpy as np
import torch.nn as nn
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_inception_net():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    return inception_model


print("Loading inception... ", end="", flush=True)
t0 = time.time()
inception = load_inception_net().cuda().eval().half()
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
        act1[start_idx:end_idx, :] = inception(images1[start_idx:end_idx])[0]
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


def fastprefid(
    images1,
    mu2,
    sigma2,
    batch_size=-1,
    preprocess=True,
    measure_time=False,
    gradient_checkpointing=False,
):
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


def train():
    # Load the weights
    G = Generator(100).to(device)
    optim = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.99))
    G.load_state_dict(
        torch.load("models/generator.pt", map_location=device), strict=False
    )

    for epoch in range(20):
        print("Beginning training at epoch %d..." % epoch)
        for i in range(100000):

            G.train()
            toggle_grad(G, True)
            optim.zero_grad()

            k = 100
            r = 10  # accumulate gradients for r=10 mini-batches
            latent_dim = 100
            with torch.cuda.amp.autocast():
                for j in range(r):
                    G_z = (torch.randn(r, latent_dim, 1, 1, device=device) + 1) / 2

                    fid = fastprefid(
                        G_z.cuda(),
                        mu_gpu,
                        sigma_gpu,
                        batch_size=torch.ones(1).requires_grad_(True) * 10,
                        gradient_checkpointing=False,
                    )
                    fid_ = fid / fid.detach().data.clone() / r
                    fid_.backward()
                    torch.cuda.empty_cache()
                    print("\r[%-5i %-2i] %-4f" % (i, j, fid.item()), end="")

            optim.step()
            optim.zero_grad()

            if (i < 100) or (i % 10 == 0):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        fixed = G_z[:64]
                        img = torchvision.utils.make_grid(fixed.detach().cpu())
                        torchvision.utils.save_image(
                            img, "samples/%d_%d.png" % (epoch, i)
                        )


if __name__ == "__main__":
    train()
