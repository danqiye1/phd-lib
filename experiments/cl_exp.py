"""
Full experiment to benchmark continual learning strategies.
"""
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize, Pad
from patterns.models import LeNet, MultiHeadLeNet
from continual.datasets import SplitMNIST
from continual.utils import train_ewc, rehearsal, pseudo_rehearsal
from patterns.utils import train_epoch, validate

from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
args = parser.parse_args()

transforms = Compose([
    ToTensor(),
    Pad(2), # For LeNet input
    Normalize(mean=(0.1307,), std=(0.3081,))
])
trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)

##################
# Naive Strategy #
##################
model = LeNet(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device(args.device_type)

error_rates = {}
val_loss = {}
for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")
    error_rates[task] = []
    val_loss[task] = []
    for epoch in tqdm(range(args.max_epoch)):
        loss = train_epoch(
                    model, trainset,
                    batch_size=args.batch_size,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device)

        # Evaluate error rate on current and previous tasks
        for task in range(trainset.get_current_task() + 1):
            vloss, verror = validate(
                        model, evalset, args.batch_size,
                        criterion=criterion, 
                        device=device
                    )
            error_rates[task].append(verror)
            val_loss[task].append(vloss)
            evalset = evalset.next_task()
        evalset = evalset.restart()

    # Progress to next task
    trainset = trainset.next_task()

bp()
    



