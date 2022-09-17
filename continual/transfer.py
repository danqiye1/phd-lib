"""
Experiment of using transfer learning on multi-head models
to mitigate catastrophic forgetting.
"""
import torch
import json
import argparse
from tqdm import tqdm
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from .datasets import SplitMNIST
from .utils import train_multihead
from patterns.models import MultiHeadLeNet
from patterns.utils import validate
from matplotlib import pyplot as plt

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
trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)

# Setup training
device = torch.device(args.device_type)
model = MultiHeadLeNet(num_classes=trainset.num_classes())

# Setup metrics collection
train_loss = {task: [] for task in range(trainset.num_tasks())}
val_loss = {task: [] for task in range(evalset.num_tasks())}
val_error = {task: [] for task in range(evalset.num_tasks())}

for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")

    if task == 0:
        epochs = config['epochs']
    else:
        epochs = 1
    
    # Train with epoch training
    for epoch in tqdm(range(epochs)):
        loss, vloss, verror = train_multihead(
                                model, trainset,
                                batch_size=config['batch_size'],
                                device=device,
                                validate_fn=validate,
                                valset=evalset)

        # Update metrics
        for key in loss:
            train_loss[key] += loss[key]
            val_loss[key] += vloss[key]
            val_error[key] += verror[key]

    # Progress to next task
    trainset = trainset.next_task()
    model.add_head(num_classes=trainset.num_classes())

plt.plot(val_error[0])
plt.savefig("results/transfer_error.jpg")

with open("results/transfer_error.json", 'w') as fp:
    json.dump(val_error, fp)
    