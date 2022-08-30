import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Pad, ToTensor, Normalize
from patterns.datasets import SplitMNIST
from patterns.models import LeNet
from patterns.utils import train_ewc, validate

from pdb import set_trace as bp

from patterns.utils.training import ewc_update

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ewc_lambda', type=float, default=0.4)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
args = parser.parse_args()

# Hyperparameters configuration
config = {
    "learning_rate": args.lr,
    "epochs": args.max_epoch,
    "batch_size": args.batch_size,
    "ewc_weight": args.ewc_lambda
}

# Setup training
model = LeNet()
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()
device = torch.device(args.device_type)

transforms = Compose([
    ToTensor(),
    Pad(2),
    Normalize(mean=(0.1307,), std=(0.3081,))
])
trainset = SplitMNIST(args.data_dir, download=True, transform=transforms)
evalset = SplitMNIST(args.data_dir, train=False, download=True, transform=transforms)

trainset[0]

# Initialize fisher matrix dict and optimum parameter dict
fisher_dict = {}
opt_param_dict = {}

for task in range(trainset.num_tasks()):
    tqdm.write(f"Training on task {trainset.get_current_task()}")
    trainloader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    evalloader = DataLoader(
        evalset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    # Train with EWC regularized weights
    for epoch in tqdm(range(config['epochs'])):
        loss = train_ewc(
                    model, trainloader, task_id=task,
                    fisher_dict=fisher_dict, 
                    opt_param_dict=opt_param_dict, 
                    ewc_weight=config['ewc_weight'],
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device)

    # Update fisher dict and optimal parameter dict
    fisher_dict, opt_param_dict = ewc_update(
                                    model, trainloader, task_id=task,
                                    fisher_dict=fisher_dict, 
                                    opt_param_dict=opt_param_dict,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    device=device)

    # Evaluate error rate on current and previous tasks
    for task in range(evalset.get_current_task() + 1):
        vloss, verror = validate(model, evalloader, criterion=criterion, device=device)
        tqdm.write(f"Evaluated task {task}")
        tqdm.write(
            f"Training loss: {loss: .3f}, Validation loss: {vloss: .3f}," 
            f"Validation error: {verror: .3f}")

    # Progress to next task
    trainset.next_task()
    evalset.next_task()