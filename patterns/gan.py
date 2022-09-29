"""
Vanilla Epoch Training on MNIST

This is for testing in training and validation functions are correct.
"""
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from patterns.models import Generator, Discriminator, weights_init
from patterns.utils import train_gan
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--z_size', type=int, default=100)
args = parser.parse_args()


# Hyperparameters configuration
config = {
    "learning_rate": args.lr,
    "epochs": args.max_epoch,
    "batch_size": args.batch_size,
}

transforms = Compose([
    ToTensor(),
    Pad(2), # For LeNet input
    Normalize(mean=(0.1307,), std=(0.3081,))
])
trainset = MNIST(args.data_dir, download=True, transform=transforms)
evalset = MNIST(args.data_dir, train=False, download=True, transform=transforms)

# Setup training
device = torch.device(args.device_type)
generator = Generator(args.z_size).apply(weights_init).to(device)
discriminator = Discriminator().apply(weights_init).to(device)

# Collect loss
generator_loss = []
discriminator_loss = []

for epoch in tqdm(range(config['epochs'])):
    gloss, dloss = train_gan(
                    generator, discriminator, trainset,
                    batch_size=config['batch_size'],
                    device=device, feature_size=args.z_size)

    tqdm.write(f"Epoch {epoch}")
    tqdm.write(
        f"Generator loss: {gloss: .3f}, Discriminator loss: {dloss: .3f}")

    generator_loss.append(gloss)
    discriminator_loss.append(dloss)

# Post training evaluation
plt.figure()
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_loss, label="G")
plt.plot(discriminator_loss, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/gan/gan_loss.jpg")

# Visualization
fixed_noise = torch.randn((64, args.z_size, 1, 1), device=args.device_type)
with torch.no_grad():
    fake_imgs = generator(fixed_noise).detach().cpu()
    img_grid = make_grid(fake_imgs)
plt.figure()
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.savefig("results/gan/gan_fake_imgs.jpg")