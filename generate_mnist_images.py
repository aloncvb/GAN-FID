import os
from torchvision.utils import save_image
import torch
from dcgan import Generator
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_generated_images(generator, latent_dim, num_images, folder_path):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, mode=0o777)

    # Generate images
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        generated_images = generator(noise)

    resize_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),  # Resize to 299x299 for FID
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        ]
    )
    # Save images
    for i, image in enumerate(generated_images):
        # Ensure the image has three channels if it is grayscale
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # Save the image
        image = resize_transform(image)
        save_image(image, os.path.join(folder_path, f"generated_{i}.png"))


latent_dim = 100
num_images = 1000
folder_path = "generated_mnist_images"  # Specify the path

generator = Generator(latent_dim).to(device)

# Load the weights
generator.load_state_dict(torch.load("models/generator.pt", map_location=device))
generator.eval()  # Set the model to evaluation mode
save_generated_images(generator, latent_dim, num_images, folder_path)
