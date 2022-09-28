"""
Vanilla Epoch Training on MNIST

This is for testing in training and validation functions are correct.
"""
import torch
import argparse
from tqdm import tqdm
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from torchvision.datasets import MNIST
from patterns.models import Generator, Discriminator, weights_init
from patterns.utils import validate, train_gan

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
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

for epoch in tqdm(range(config['epochs'])):
    gloss, dloss = train_gan(
                    generator, discriminator, trainset,
                    batch_size=config['batch_size'],
                    device=device, feature_size=args.z_size)

    tqdm.write(f"Epoch {epoch}")
    tqdm.write(
        f"Generator loss: {gloss: .3f}, Discriminator loss: {dloss: .3f}")