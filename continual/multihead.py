"""
Experiment of using transfer learning on multi-head models
to mitigate catastrophic forgetting.
"""
import torch
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from .datasets import SplitMNIST, PermutedMNIST
from .utils import train_multihead, plot_task_error
from .models import MultiHead
from patterns.utils import validate, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--dataset', type=str, default="SplitMNIST", choices=['SplitMNIST', 'PermutedMNIST'])
parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'mlp'])
args = parser.parse_args()

# Hyperparameters configuration
config = {
    "learning_rate": args.lr,
    "epochs": args.max_epoch,
    "batch_size": args.batch_size,
}

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

if args.dataset == "SplitMNIST":
    trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
    evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)
elif args.dataset == "PermutedMNIST":
    trainset = PermutedMNIST(args.data_dir, download=True, transform=transforms)
    evalset = PermutedMNIST(args.data_dir, train=False, download=True, transform=transforms)

# Setup training
device = torch.device(args.device_type)
model = MultiHead(device=device, benchmark=args.dataset, architecture=args.model).to(device)

# Setup metrics collection
train_loss = {task: [] for task in range(trainset.num_tasks())}
val_loss = {task: [] for task in range(evalset.num_tasks())}
val_error = {task: [] for task in range(evalset.num_tasks())}

# Data structure for recording iteration boundaries for each task.
boundaries = [0 for _ in range(evalset.num_tasks())]

for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")
    model.add_head(num_classes=trainset.num_classes())
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

    # Record number of iterations for each task
    boundaries[task] = len(train_loss[task]) if task == 0 else (len(train_loss[task]) + boundaries[task - 1])

    # Progress to next task
    trainset = trainset.next_task()

testset = SplitMNIST(
            args.data_dir,
            train=False, 
            transform=transforms, 
            tasks=[[0,1,2,3,4,5,6,7,8,9]])

plt.figure("confusion")
confusion_display = confusion_matrix(model, testset, device=device)
confusion_display.plot()
plt.savefig("results/multihead_confusion.jpg")


with open("results/multihead_error.json", 'w') as fp:
    json.dump(val_error, fp)

with open("results/multihead_boundaries.json", "w") as fp:
    json.dump(boundaries, fp)

plot_task_error(0, val_error, boundaries=boundaries, strategy="results/multihead")
    