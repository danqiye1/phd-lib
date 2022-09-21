"""
Script for plotting the results.
"""
import os
import json
import argparse
from matplotlib import pyplot as plt
from .utils import plot_task_error

from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--epochs', type=str, default='5', choices=['3', '5'])
parser.add_argument('--benchmark', type=str, default='splitmnist', choices=['splitmnist', 'permutedmnist'])
parser.add_argument('--strategy', type=str, default='all', 
                    choices=['all', 'ewc', 'pseudo', 'rehearsal', 'multihead'])
args = parser.parse_args()

# Preprocess path to results
results_dir = f'{args.results_dir}/{args.benchmark}/{args.epochs}'

# Open boundaries file
with open(f'{results_dir}/boundaries.json', 'r') as fp:
    boundaries = json.load(fp)

num_tasks = len(boundaries)

if args.strategy == 'all':
    for strategy in  ('ewc', 'pseudo', 'rehearsal', 'multihead'):
        filepath = os.path.join(results_dir, f'{strategy}_error.json')

        with open(filepath, 'r') as fp:
            result = json.load(fp)

        for task in result.keys():
            plot_task_error(task, result, boundaries=boundaries, save=False, strategy=strategy)
else:
    filepath = os.path.join(results_dir, f'{args.strategy}_error.json')

    with open(filepath, 'r') as fp:
        result = json.load(fp)

    for task in result.keys():
        plot_task_error(task, result, boundaries=boundaries, save=False, strategy=args.strategy)


for task in range(num_tasks):
    plt.figure(task + 1)
    plt.savefig("")