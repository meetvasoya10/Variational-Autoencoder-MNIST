import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vae import VAE

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=20).to(device)
vae.load_state_dict(torch.load("vae_mnist.pt", map_location=device))
vae.eval()

# Load test data
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Encode all test images to latent space
latent_vectors = []
labels = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        mu, logvar = vae.encode(data.view(-1, 28*28))
        z = vae.reparameterize(mu, logvar)
        latent_vectors.append(z.cpu())
        labels.append(target.cpu())

latent_vectors = torch.cat(latent_vectors, dim=0)
labels = torch.cat(labels, dim=0)

# Pick one latent vector per digit 0–9
digit_latents = []
for digit in range(10):
    idx = (labels == digit).nonzero(as_tuple=True)[0][0]
    z_digit = latent_vectors[idx].unsqueeze(0).to(device)
    digit_latents.append(z_digit)

# Decode each latent to an image
decoded_images = []
with torch.no_grad():
    for z in digit_latents:
        img = vae.decode(z).cpu().view(28, 28)
        decoded_images.append(img)

# Plot with colors
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    img_np = decoded_images[i].numpy()
    colormap = cm.get_cmap('inferno') 
    img_colored = colormap(img_np)[:, :, :3]  # RGB only
    ax.imshow(img_colored)
    ax.axis('off')
    ax.set_title(str(i), fontsize=14)

plt.suptitle("VAE Generated Digits 0–9", fontsize=18)
plt.tight_layout()
plt.show()
