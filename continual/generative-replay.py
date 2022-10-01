"""
Experiment of using transfer learning on multi-head models
to mitigate catastrophic forgetting.
"""
import torch
import json
import argparse
from .models import Scholar, MLP
from patterns.models import LeNet, Generator, Discriminator, weights_init
from patterns.utils import validate
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from .datasets import SplitMNIST, PermutedMNIST
from tqdm import tqdm
from copy import deepcopy
from .utils import plot_task_error

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--dataset', type=str, default="SplitMNIST", choices=['SplitMNIST', 'PermutedMNIST'])
parser.add_argument('--z_size', type=int, default=100)
parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'mlp'])
args = parser.parse_args()

# Hyperparameters configuration
config = {
    "learning_rate": args.lr,
    "epochs": args.max_epoch,
    "batch_size": args.batch_size,
}

# Setup training
solver = LeNet() if args.model == 'lenet' else MLP()
generator = Generator(args.z_size).apply(weights_init)
discriminator = Discriminator().apply(weights_init)
device = torch.device(args.device_type)

if args.model == 'lenet':
    transforms = Compose([
        ToTensor(),
        Pad(2),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
else:
    transforms = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
if args.dataset == 'SplitMNIST':
    trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
    evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)
elif args.dataset == "PermutedMNIST":
    trainset = PermutedMNIST(args.data_dir, download=True, transform=transforms)
    evalset = PermutedMNIST(args.data_dir, train=False, download=True, transform=transforms)

# Setup metrics collection
train_loss = {task: [] for task in range(trainset.num_tasks())}
val_loss = {task: [] for task in range(evalset.num_tasks())}
val_error = {task: [] for task in range(evalset.num_tasks())}

# Data structure for recording iteration boundaries for each task.
boundaries = [0 for _ in range(evalset.num_tasks())]

# List of scholars for each task
scholars = []
old_scholar = None
for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")

    if task == 0:
        # For first task, train normally without mixing data and replay
        epochs = config['epochs']
        new_scholar = Scholar(
                        deepcopy(generator), 
                        deepcopy(discriminator), 
                        deepcopy(solver),
                        task_id=task,
                        device=device)
        mix_ratio = 1
    else:
        epochs = 1
        new_scholar = Scholar(
                        deepcopy(old_scholar.generator), 
                        deepcopy(old_scholar.discriminator), 
                        deepcopy(old_scholar.solver),
                        task_id=task,
                        device=device)
        mix_ratio = 0.5

    # Train generator
    for epoch in tqdm(range(epochs)):
        gloss, dloss = new_scholar.train_generator(trainset)

    # Train solver
    for epoch in tqdm(range(epochs)):
        loss, vloss, verror = new_scholar.train_solver(
                                trainset, old_scholar, 
                                device=device,
                                mix_ratio=mix_ratio,
                                validate_fn=validate, 
                                valset=evalset)

        # Update metrics
        for key in loss:
            train_loss[key] += loss[key]
            val_loss[key] += vloss[key]
            val_error[key] += verror[key]

    # Add a scholar for this task
    scholars.append(new_scholar)
    old_scholar = new_scholar
    

    # Record number of iterations for each task
    boundaries[task] = len(train_loss[task]) if task == 0 else (len(train_loss[task]) + boundaries[task - 1])

    # Progress to next task
    trainset = trainset.next_task()

with open("results/gr_error.json", 'w') as fp:
    json.dump(val_error, fp)

with open("results/gr_boundaries.json", "w") as fp:
    json.dump(boundaries, fp)