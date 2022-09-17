"""
SplitMNIST benchmark using EWC from Avalanche
"""
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Pad, Normalize, ToTensor
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.training.supervised import Replay
from avalanche.models import LeNet5
from avalanche.logging import InteractiveLogger, WandBLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin

from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ewc_lambda', type=float, default=0.4)
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--device_type', type=str, default="cuda:0", choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--dataset', type=str, default="SplitMNIST", choices=['SplitMNIST', 'PermutedMNIST'])
parser.add_argument('--log_folder', type=str, default="./results/rehearsal")
args = parser.parse_args()

transforms = Compose([
    ToTensor(),
    Pad(2),
    Normalize(mean=(0.1307,), std=(0.3081,))
])
benchmark = SplitMNIST(
                n_experiences=5, 
                return_task_id=False, 
                train_transform=transforms, 
                eval_transform=transforms)

model = LeNet5(n_classes=benchmark.n_classes, input_channels=1)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

loggers = [
    InteractiveLogger(), 
    WandBLogger(project_name="ContinualML", run_name="Rehearsal"),
    CSVLogger(log_folder=args.log_folder)
]


eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=loggers,
    benchmark=benchmark
)


tqdm.write("Starting experiment...")
results = []
num_exp = len(benchmark.train_stream)
for i, exp in tqdm(enumerate(benchmark.train_stream), total=num_exp):

    train_epochs = args.max_epoch if i == 0 else 1

    strategy = Replay(
            model, optimizer, criterion,
            train_mb_size=args.batch_size,
            train_epochs=train_epochs,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            eval_every=1)

    res = strategy.train(exp, eval_streams=[benchmark.test_stream])
    tqdm.write("Training completed!")
    tqdm.write("Computing accuracy on test set...")
    results.append(strategy.eval(benchmark.test_stream))