"""
Plotting utilities
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_task_error(
        task_num, val_error, 
        boundaries=[], 
        save=True,
        strategy='example',
        benchmark='split'
    ):
    # Hard-code for QE. Next time make more generic
    plt.figure(task_num)
    if benchmark == 'split':
        class_labels = [(0,1), (2,3), (4,5), (6,7), (8,9)]
        label = class_labels[int(task_num)]
        plt.title(f'Error on Class Label {label}')
    else:
        plt.title(f"Error on Permuted Task {task_num}")
    
    plt.xlabel("iterations")
    plt.ylabel("error")
    for x in boundaries:
        plt.vlines(x, ymin=0, ymax=1, color='grey', linestyles="dashed", alpha=0.5)
    
    # Create the x-values from xstart to xend
    # xend is always the total length of the first task
    # xstart is whatever boundary of the previous task
    xstart = (boundaries[int(task_num) - 1] if int(task_num) else 0)
    xend = len(val_error['0'])
    xaxis = np.arange(xstart, xend)
    plt.plot(xaxis, val_error[task_num], label=strategy)
    
    if save:
        plt.savefig(f'{strategy}.jpg')