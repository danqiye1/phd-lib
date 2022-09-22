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
    ):
    plt.figure(task_num)
    plt.title(f'Error for Task {task_num}')
    plt.xlabel("iterations")
    plt.ylabel("error")
    for x in boundaries:
        plt.vlines(x, ymin=0, ymax=1, color='r', linestyles="dashed")
    
    # Create the x-values from xstart to xend
    # xend is always the total length of the first task
    # xstart is whatever boundary of the previous task
    xstart = (boundaries[int(task_num) - 1] if int(task_num) else 0)
    xend = len(val_error[0])
    xaxis = np.arange(xstart, xend)
    plt.plot(xaxis, val_error[task_num], label=strategy)
    
    if save:
        plt.savefig(f'{strategy}.jpg')