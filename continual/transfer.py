"""
Experiment of using transfer learning on multi-head models
to mitigate catastrophic forgetting.
"""
import torch
import argparse
from tqdm import tqdm
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from .datasets import SplitMNIST
from .utils import train_epoch
from patterns.models import MultiHeadLeNet
from patterns.utils import validate

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

for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")
    # Train with epoch training
    for epoch in tqdm(range(config['epochs'])):
        loss = train_epoch(
                    model, trainset,
                    batch_size=config['batch_size'],
                    device=device)

    # Evaluate error rate on current and previous tasks
    for task in range(trainset.get_current_task() + 1):
        vloss, verror = validate(
                            model, evalset, config['batch_size'],
                            device=device
                        )
        tqdm.write(f"Evaluated task {task}")
        tqdm.write(
            f"Training loss: {loss: .3f}, Validation loss: {vloss: .3f}, " 
            f"Validation error: {verror: .3f}")
        evalset = evalset.next_task()

    # Progress to next task
    trainset = trainset.next_task()
    evalset = evalset.restart()
    model.add_head(num_classes=trainset.num_classes())
    