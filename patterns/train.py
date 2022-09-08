"""
Vanilla Epoch Training on MNIST

This is for testing in training and validation functions are correct.
"""
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from torchvision.datasets import MNIST
from patterns.models import LeNet
from patterns.utils import validate, train_epoch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
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
criterion = torch.nn.CrossEntropyLoss()
device = torch.device(args.device_type)
model = LeNet(10).to(device)

evalloader = DataLoader(
                    evalset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=4
                )

for epoch in tqdm(range(config['epochs'])):
    loss = train_epoch(
                model, trainset,
                batch_size=config['batch_size'],
                criterion=criterion,
                device=device)

    vloss, verror = validate(model, evalloader, criterion=criterion, device=device)
    tqdm.write(f"Epoch {epoch}")
    tqdm.write(
        f"Training loss: {loss: .3f}, Validation loss: {vloss: .3f}, " 
        f"Validation error: {verror: .3f}")