"""
Demo of Rehearsal Mechanism
"""
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from .datasets import SplitMNIST, PermutedMNIST
from .utils import rehearsal
from patterns.models import LeNet
from patterns.utils import validate

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--dataset', type=str, default="SplitMNIST", choices=['SplitMNIST', 'PermutedMNIST'])
args = parser.parse_args()

# Hyperparameters configuration
config = {
    "learning_rate": args.lr,
    "epochs": args.max_epoch,
    "batch_size": args.batch_size,
}

# Setup training
model = LeNet()
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()
device = torch.device(args.device_type)

transforms = Compose([
    ToTensor(),
    Pad(2), # For LeNet input
    Normalize(mean=(0.1307,), std=(0.3081,))
])
if args.dataset == 'SplitMNIST':
    trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
    evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)
elif args.dataset == "PermutedMNIST":
    trainset = PermutedMNIST(args.data_dir, download=True, transform=transforms)
    evalset = PermutedMNIST(args.data_dir, train=False, download=True, transform=transforms)

for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")

    # Train with rehearsal strategy
    for epoch in tqdm(range(config['epochs'])):
        loss = rehearsal(
                    model, trainset,
                    batch_size=config['batch_size'],
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device)

    # Evaluate error rate on current and previous tasks
    for task in range(evalset.num_tasks()):
        evalset = evalset.go_to_task(task)
        vloss, verror = validate(
                    model, evalset, config['batch_size'],
                    criterion=criterion, 
                    device=device
                )
        tqdm.write(f"Evaluated task {task}")
        tqdm.write(
            f"Training loss: {loss: .3f}, Validation loss: {vloss: .3f}, " 
            f"Validation error: {verror: .3f}")

    # Progress to next task
    trainset = trainset.next_task()
    